import os
import random
import torch
import numpy as np
from beamformer.utils import count_parameters, scale_in_last_dim, load_config, get_log_path,namespace_to_dict
from beamformer.modules import TransformerModel, get_scheduler_by_type
from beamformer.dataset import load_data_process, load_datasets
from beamformer.weight_generator import PredefinedGenerator, ParametricGenerator, transform_weights
from accelerate import Accelerator
import os
import numpy as np
import pprint
import glob

class Trainer:
    def __init__(self, config):
        
        self.config = config

        self.log_path = get_log_path()

        self.M = config.dataset.M
        self.N = config.dataset.N

        self.accelerator = Accelerator(gradient_accumulation_steps=1)

        self.device = self.accelerator.device

        self._set_seeds()
        
        self.train_dataset, self.test_dataset = load_datasets(config)
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_size=config.training.batch_size, 
            shuffle=True,
            num_workers=config.training.num_workers,
            drop_last=True,
        )

    def log(self, message):
        if self.accelerator.is_main_process:
            print(message)
            with open(self.log_path, 'a') as f:
                f.write(message + '\n')
    
    def is_generator_ml(self):
        return self.config.training.scheme in ["co-train"]

    def initialize_generator(self):
        sample_num = self.config.assumption.sample_num
        if self.is_generator_ml():
            if self.config.generator.type == "PARAM":
                generator = ParametricGenerator(sample_num, self.M, self.N)
        elif self.config.training.scheme in "pre-defined":
            generator = PredefinedGenerator(
                self.config.generator.M_act, self.config.generator.N_act,
                sample_mode=self.config.generator.type,
                sample_num=sample_num,
                M_base=self.M,
                N_base=self.N,
                start_freq=self.config.dataset.start_freq,
                end_freq=self.config.dataset.end_freq,
                angle_steps_theta=self.config.assumption.angle_steps_theta,
                angle_steps_phi=self.config.assumption.angle_steps_phi,
                freq_num=self.config.dataset.freq_num,
                max_theta=self.config.dataset.max_theta,
                d_row=self.config.dataset.d_row,
                d_col=self.config.dataset.d_col,
                device = self.device,
            )
        else:
            raise ValueError(f"Unsupported training scheme: {self.config.training.scheme}")


        if self.is_generator_ml():
            if self.config.generator.generator_pretrained_model:
                generator.load_state_dict(torch.load(self.config.generator.generator_pretrained_model, weights_only=True))
                self.log(f"Generator loaded from {self.config.generator.generator_pretrained_model}")
            if self.accelerator.is_main_process:
                count_parameters(generator, "Generator")
        return generator

    def initialize_estimator(self):
        cfg = self.config.estimator  
        estimator = TransformerModel(estimator_config=cfg)
        if cfg.estimator_pretrained_model:
            estimator.load_state_dict(torch.load(cfg.estimator_pretrained_model, weights_only=True))
            self.log(f"Estimator loaded from {cfg.estimator_pretrained_model}")
        if self.accelerator.is_main_process:
            count_parameters(estimator, "Estimator")
        return estimator.to(self.device)

    def save_model(self, model, name):
        path = os.path.join(self.config.training.model_save_path, f"{name}.pth")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        model_files = sorted(
            glob.glob(os.path.join(self.config.training.model_save_path, "*.pth")),
            key=os.path.getmtime
        )
        if len(model_files) >= 2:
            oldest_file = model_files[0]
            os.remove(oldest_file)
            self.log(f"Deleted old model: {oldest_file}")

        self.accelerator.save(model.state_dict(), path)
        self.log(f"Model saved at {path}")

    def _check_model_exist(self, path, size_kb=10):
        if not os.path.exists(path) or not os.path.isdir(path):
            return False

        total_size = 0
        for root, _, files in os.walk(path):
            for f in files:
                try:
                    fp = os.path.join(root, f)
                    total_size += os.path.getsize(fp)
                except OSError:
                    continue  

        return total_size > size_kb * 1024

    def _forward_pass_and_loss(self, csi, generator, estimator, dp, z, 
                            require_grad_generator: bool, 
                            require_grad_estimator: bool):
        if self.is_generator_ml():
            for p in generator.parameters():
                p.requires_grad = require_grad_generator
        for p in estimator.parameters():
            p.requires_grad = require_grad_estimator

        if self.config.training.scheme == "pre-defined":
            weights_out, _ = generator.generate(batch_size=self.config.training.batch_size)
        else:
            raw_weights = generator(z)
            weights_out, _ = transform_weights(raw_weights)

        sample_pos_enc = dp.generate_sample_position_encoding(weights_out)
        query_pos_enc = dp.generate_query_position_encoding(batch_size = self.config.training.batch_size)
        sample_rss = dp.generate_sample_rss(csi, weights_out)
        query_rss = dp.generate_query_rss(csi)

        scale = torch.max(sample_rss, dim=1, keepdim=True).values
        sample_rss /= scale
        query_rss /= scale

        sample_rss = sample_rss.to(dtype=torch.float32)
        query_rss = query_rss.to(dtype=torch.float32)
        sample_pos_enc = sample_pos_enc.to(dtype=torch.float32)
        query_pos_enc = query_pos_enc.to(dtype=torch.float32)

        label_gt = query_rss
        label_gt, _ = scale_in_last_dim(label_gt)

        pred_rss, _ = estimator(sample_rss, sample_pos_enc, query_pos_enc)

        loss, as_loss_db = self._loss_fun(label_gt, pred_rss)
        return loss, as_loss_db


    def _save_best_model(self, loss, epoch, batch_idx, estimator, generator):

        if not hasattr(self, "best_loss"):
            self.best_loss = float("0.0005")


        if loss < self.best_loss:
            self.best_loss = loss
            self.log(f"New best model at Epoch {epoch+1}, Batch {batch_idx} with loss {self.best_loss:.6f}")

            estimator_to_save = self.accelerator.unwrap_model(estimator)
            if self.is_generator_ml():
                generator_to_save = self.accelerator.unwrap_model(generator)

            if self.accelerator.is_main_process:
                self.save_model(estimator_to_save, "best_estimator")
                if self.is_generator_ml():
                    self.save_model(generator_to_save, "best_generator")

                record_path = os.path.join(self.config.training.model_save_path, "best_record.txt")
                with open(record_path, "w") as f:
                    f.write(f"Best loss: {self.best_loss:.6f} at Epoch {epoch+1}, Batch {batch_idx}\n")

    

    def _loss_fun(self, tensor_gt, tensor_pred):

        mse = torch.nn.MSELoss()

        mse_loss_as = mse(tensor_gt, tensor_pred)
        mse_loss_as_db = 10 * torch.log10(mse_loss_as + 1e-16)

        return mse_loss_as, mse_loss_as_db
    
    def train(self):
        
        if self._check_model_exist(self.config.training.model_save_path):
            raise FileExistsError("You Use An Existing Folder as Save Path")
        generator = self.initialize_generator()
        estimator = self.initialize_estimator()

        scheme = self.config.training.scheme
        self.log(f"Training scheme: {scheme}")

        estimator.train()


        if self.config.estimator.optimizer.lower() == "adamw":
            optimizer_estimator = torch.optim.AdamW(
                estimator.parameters(), lr=self.config.training.lr_estimator, weight_decay=0.01
            )

        else:
            raise ValueError(f"Unsupported optimizer type: {self.config.estimator.optimizer}")


        total_steps = self.config.training.epochs * len(self.train_dataloader) / self.config.training.gpu_num
        warmup_steps = int(self.config.training.warmup_ratio * total_steps)

        scheduler_estimator = get_scheduler_by_type(
            self.config.estimator.estimator_lr_scheduler_type,
            optimizer_estimator,
            warmup_steps,
            total_steps,
        )

        if self.is_generator_ml():
            generator.train()
            optimizer_generator = torch.optim.AdamW(
                generator.parameters(), lr=self.config.training.lr_generator, weight_decay=0.01
            )
            
            scheduler_generator = get_scheduler_by_type(
                self.config.generator.generator_lr_scheduler_type,
                optimizer_generator,
                warmup_steps,
                total_steps,
            )

            train_dataloader, optimizer_generator, optimizer_estimator, generator, estimator = self.accelerator.prepare(
                self.train_dataloader, optimizer_generator, optimizer_estimator, generator, estimator
            )
        else:
            train_dataloader, estimator, optimizer_estimator = self.accelerator.prepare(
                self.train_dataloader, estimator, optimizer_estimator
            )

        dp = load_data_process(self.config, device=self.device)
        
        for epoch in range(self.config.training.epochs):
            epoch_loss = 0.0
            for batch_idx, (csi, csi_path) in enumerate(train_dataloader):

                if scheme == "co-train":
                    loss, as_loss_db = self._co_train_logic(csi, generator, estimator, dp, optimizer_estimator, scheduler_estimator, optimizer_generator, scheduler_generator)
                elif scheme == "pre-defined":
                    loss, as_loss_db = self._pre_defined_logic(csi, generator, estimator, dp, optimizer_estimator, scheduler_estimator)
                epoch_loss += loss
                current_lr_estimator = scheduler_estimator.get_last_lr()[0]

                current_lr_estimator = scheduler_estimator.get_last_lr()[0]
                
                log_msg = f"Epoch [{epoch+1}] Batch [{batch_idx}] | EstLR: {current_lr_estimator:.2e}"

                if self.is_generator_ml() and scheduler_generator is not None:
                    current_lr_generator = scheduler_generator.get_last_lr()[0]
                    log_msg += f" | GenLR: {current_lr_generator:.2e}"

                log_msg += f" | Loss: {loss:.6f} | AS Loss (dB): {as_loss_db:.3f}"
                self.log(log_msg)



            avg_loss = epoch_loss / (len(self.train_dataloader) * self.config.training.gpu_num)
            self.log(f"Epoch [{epoch+1}] Average Loss: {avg_loss:.6f}")

            self.accelerator.wait_for_everyone()

            if self.accelerator.is_main_process:  
                epoch_tag = f"epoch_final" if (epoch + 1) == self.config.training.epochs else f"epoch{epoch + 1}"
                estimator_to_save = self.accelerator.unwrap_model(estimator)
                self.save_model(estimator_to_save, f"estimator_{epoch_tag}")
                if self.is_generator_ml():
                    generator_to_save = self.accelerator.unwrap_model(generator)
                    self.save_model(generator_to_save, f"generator_{epoch_tag}")
        print("Training complete.")
    
    def _co_train_logic(self,csi, generator, estimator, dp, optimizer_estimator, scheduler_estimator, optimizer_generator, scheduler_generator):
        z_dim = self.config.assumption.sample_num * self.config.dataset.M * self.config.dataset.N
        z = torch.randn(self.config.training.batch_size, z_dim).to(self.device)
        loss, as_loss_db = self._forward_pass_and_loss(
            csi, generator, estimator, dp, z,
            require_grad_generator=True, 
            require_grad_estimator=True    
        )
        optimizer_estimator.zero_grad()
        optimizer_generator.zero_grad()
        self.accelerator.backward(loss)
        self.accelerator.clip_grad_norm_(estimator.parameters(), 0.5)
        optimizer_estimator.step()
        optimizer_generator.step()
        scheduler_estimator.step()
        scheduler_generator.step()
        return loss.item(), as_loss_db.item()

    def _pre_defined_logic(self,csi, generator, estimator, dp, optimizer_estimator, scheduler_estimator):
        loss, as_loss_db = self._forward_pass_and_loss(
            csi, generator, estimator, dp, None,
            require_grad_generator=None, # pre_defined generator does not have this attribute
            require_grad_estimator=True    
        )
        optimizer_estimator.zero_grad()
        self.accelerator.backward(loss)
        self.accelerator.clip_grad_norm_(estimator.parameters(), 0.5)
        optimizer_estimator.step()
        scheduler_estimator.step()
        return loss.item(), as_loss_db.item()
    
    def _set_seeds(self):
        seed = self.config.training.random_seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if self.accelerator.is_main_process:
            config_dict = namespace_to_dict(self.config)
            config_str = pprint.pformat(config_dict, indent=4, width=100)
            self.log("========== Configuration ==========\n" + config_str + "\n==================================")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Training Script with Config Support")
    parser.add_argument("--config", type=str, default=None, 
                        help="name of config file under configs")

    args = parser.parse_args()

    config_name = args.config

    print(f"You're using config: {config_name}")

    config = load_config(config_name)

    # Assuming you have a Trainer class defined elsewhere
    trainer = Trainer(config)
    trainer.train()
