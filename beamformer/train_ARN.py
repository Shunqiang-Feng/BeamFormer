
from beamformer.train import Trainer as Trainer_basic

import torch
from beamformer.utils import count_parameters, scale_in_last_dim, load_config
from beamformer.modules import get_scheduler_by_type, AmplitudeRecoveryNetwork
from beamformer.dataset import load_data_process
from beamformer.weight_generator import transform_weights


class Trainer(Trainer_basic):

    def init_arn(self):
        cfg = self.config.arn_model
        sample_num = self.config.assumption.sample_num
        arn_model = AmplitudeRecoveryNetwork(sample_num=sample_num, d_model=cfg.hidden_dims)
        if self.accelerator.is_main_process:
            count_parameters(arn_model, "ARN")
        return arn_model

    def _forward_pass_and_loss(self, csi, generator, arn_model, dp, z):
        with torch.no_grad():
            raw_weights = generator(z)
            weights_out, _ = transform_weights(raw_weights)

        

        sample_rss = dp.generate_sample_rss(csi, weights_out)
        query_rss = dp.generate_query_rss(csi)

        scale = torch.max(sample_rss, dim=1, keepdim=True).values
        sample_rss /= scale
        query_rss /= scale

        sample_rss = sample_rss.to(dtype=torch.float32)
        query_rss = query_rss.to(dtype=torch.float32)

        _, peak_scale = scale_in_last_dim(query_rss)
        peak_scale_rec = 1.0 / peak_scale
        label = peak_scale_rec.repeat(1, self.config.assumption.sample_num)

        pred = arn_model(sample_rss)
        loss, _ = self._loss_fun(pred, label)

        return loss
    
    def train(self):
        if self._check_model_exist(self.config.training.model_save_path):
            raise FileExistsError("You Use An Existing Folder as Save Path")
        
        generator = self.initialize_generator()
        if self.is_generator_ml():
            generator = generator.to(self.device)
        arn_model = self.init_arn()

        arn_model.train()
        if self.is_generator_ml():
            generator.eval()

        optimizer = torch.optim.AdamW(
            arn_model.parameters(), lr=self.config.training.learning_rate, weight_decay=0.01
        )

        total_steps = self.config.training.epochs * len(self.train_dataloader) / self.config.training.gpu_num
        warmup_steps = int(self.config.training.warmup_ratio * total_steps)

        scheduler = get_scheduler_by_type(
            optimizer,
            warmup_steps,
            total_steps,
        )



        train_dataloader, arn_model, optimizer = self.accelerator.prepare(
            self.train_dataloader, arn_model, optimizer
        )

        dp = load_data_process(self.config, device=self.device)

        for epoch in range(self.config.training.epochs):
            for batch_idx, (csi, _) in enumerate(train_dataloader):

                z_dim = self.config.assumption.sample_num * self.config.dataset.M * self.config.dataset.N
                z = torch.randn(self.config.training.batch_size, z_dim).to(self.device)
                
                loss = self._forward_pass_and_loss(csi, generator, arn_model, dp, z)
                optimizer.zero_grad()
                self.accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                self.log(f"Epoch {epoch + 1} | Batch {batch_idx} | Loss: {loss:.6f}")

            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                epoch_tag = f"epoch_final" if (epoch + 1) == self.config.training.epochs else f"epoch{epoch + 1}"
                model_to_save = self.accelerator.unwrap_model(arn_model)
                self.save_model(model_to_save, f"model_{epoch_tag}")
        
        print("Training complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Training Script with Config Support")
    parser.add_argument("--config", type=str, default=None, 
                        help="name of config file under configs")

    args = parser.parse_args()

    config_name = args.config

    print(f"You're using config: {config_name}")

    config = load_config(config_name)  # loads from configs/<config_name>.py

    # Assuming you have a Trainer class defined elsewhere
    trainer = Trainer(config)
    trainer.train()
