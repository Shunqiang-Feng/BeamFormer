import os
import random
import torch
import numpy as np
from beamformer.utils import count_parameters, scale_in_last_dim, load_config, get_log_path,namespace_to_dict
from beamformer.modules import TransformerModel, CNNGeneratorModel, MLPAutoEncoderModel, get_scheduler_by_type, ArrayAdapter
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

    def has_array_adapter(self):
        return hasattr(self.config, 'array_adapter') and self.config.array_adapter is not None

    def initialize_array_adapter(self):
        cfg = self.config.array_adapter
        adapter = ArrayAdapter(cfg.input_array_sampling, cfg.output_array_sampling)
        if getattr(cfg, 'pretrained_model', None):
            adapter.load_state_dict(torch.load(cfg.pretrained_model, weights_only=True))
            self.log(f"ArrayAdapter loaded from {cfg.pretrained_model}")
        if self.accelerator.is_main_process:
            count_parameters(adapter, "ArrayAdapter")
        return adapter.to(self.device)

    def initialize_estimator(self):
        cfg = self.config.estimator
        estimator_type = cfg.type.lower()
        if estimator_type == "cnn_generator":
            cfg.angle_steps_phi   = self.config.assumption.angle_steps_phi
            cfg.angle_steps_theta = self.config.assumption.angle_steps_theta
            cfg.sample_num        = self.config.assumption.sample_num
            cfg.max_theta         = self.config.dataset.max_theta
            estimator = CNNGeneratorModel(estimator_config=cfg)
        elif estimator_type == "mlp_ae":
            cfg.angle_steps_phi   = self.config.assumption.angle_steps_phi
            cfg.angle_steps_theta = self.config.assumption.angle_steps_theta
            estimator = MLPAutoEncoderModel(estimator_config=cfg)
        else:
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
        self.accelerator.save(model.state_dict(), path)
        self.log(f"Model saved at {path}")

    def save_checkpoint(self, epoch, estimator, generator, optimizer_estimator, scheduler_estimator,
                        optimizer_generator=None, scheduler_generator=None):
        """Save full training state checkpoint for resuming."""
        save_dir = self.config.training.model_save_path
        os.makedirs(save_dir, exist_ok=True)

        # Keep only the 2 most recent checkpoints
        ckpt_files = sorted(
            glob.glob(os.path.join(save_dir, "checkpoint_epoch*.pt")),
            key=os.path.getmtime
        )
        while len(ckpt_files) >= 2:
            os.remove(ckpt_files[0])
            self.log(f"Deleted old checkpoint: {ckpt_files[0]}")
            ckpt_files = ckpt_files[1:]

        ckpt = {
            'epoch': epoch,
            'total_epochs': self.config.training.epochs,
            'estimator_state_dict': self.accelerator.unwrap_model(estimator).state_dict(),
            'optimizer_estimator_state_dict': optimizer_estimator.state_dict(),
            'scheduler_estimator_state_dict': scheduler_estimator.state_dict(),
            'best_loss': getattr(self, 'best_loss', float('inf')),
        }
        if self.is_generator_ml() and optimizer_generator is not None:
            ckpt['generator_state_dict'] = self.accelerator.unwrap_model(generator).state_dict()
            ckpt['optimizer_generator_state_dict'] = optimizer_generator.state_dict()
            ckpt['scheduler_generator_state_dict'] = scheduler_generator.state_dict()
        if self.has_array_adapter():
            ckpt['array_adapter_state_dict'] = self.accelerator.unwrap_model(self.array_adapter).state_dict()

        ckpt_path = os.path.join(save_dir, f"checkpoint_epoch{epoch + 1}.pt")
        self.accelerator.save(ckpt, ckpt_path)
        self.log(f"Checkpoint saved at {ckpt_path}")

    def load_checkpoint(self, ckpt, estimator, generator, optimizer_estimator, scheduler_estimator,
                        optimizer_generator=None, scheduler_generator=None):
        """Restore model weights, optimizer, scheduler, and epoch from a pre-loaded checkpoint dict.
        total_epochs must already be synced to config before calling this. Returns start_epoch."""
        self.accelerator.unwrap_model(estimator).load_state_dict(ckpt['estimator_state_dict'])
        optimizer_estimator.load_state_dict(ckpt['optimizer_estimator_state_dict'])
        scheduler_estimator.load_state_dict(ckpt['scheduler_estimator_state_dict'])

        if self.is_generator_ml() and 'generator_state_dict' in ckpt:
            self.accelerator.unwrap_model(generator).load_state_dict(ckpt['generator_state_dict'])
            optimizer_generator.load_state_dict(ckpt['optimizer_generator_state_dict'])
            scheduler_generator.load_state_dict(ckpt['scheduler_generator_state_dict'])

        if self.has_array_adapter() and 'array_adapter_state_dict' in ckpt:
            self.accelerator.unwrap_model(self.array_adapter).load_state_dict(ckpt['array_adapter_state_dict'])

        if 'best_loss' in ckpt:
            self.best_loss = ckpt['best_loss']

        start_epoch = ckpt['epoch'] + 1
        self.log(f"Checkpoint loaded: resuming from epoch {start_epoch + 1} / {self.config.training.epochs}")
        return start_epoch

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
            grid_indices = getattr(generator, '_last_grid_indices', None)
        else:
            raw_weights = generator(z)
            weights_out, _ = transform_weights(raw_weights)
            grid_indices = None

        query_valid_num = getattr(self.config.assumption, 'query_valid_num', None)
        if query_valid_num is not None:
            query_indices = torch.randperm(self.config.assumption.angle_spectrum_length, device=self.device)[:query_valid_num]
        else:
            query_indices = None

        sample_pos_enc = dp.generate_sample_position_encoding(weights_out)
        sample_rss = dp.generate_sample_rss(csi, weights_out)
        query_rss = dp.generate_query_rss(csi, indices=query_indices)

        scale = torch.max(sample_rss, dim=1, keepdim=True).values
        sample_rss /= scale
        query_rss /= scale

        sample_rss = sample_rss.to(dtype=torch.float32)
        query_rss = query_rss.to(dtype=torch.float32)
        sample_pos_enc = sample_pos_enc.to(dtype=torch.float32)

        if self.has_array_adapter():
            sample_pos_enc = self.array_adapter(sample_pos_enc)
            batch_size = sample_rss.shape[0]
            raw = self.query_pos_enc_raw[:, query_indices, :] if query_indices is not None else self.query_pos_enc_raw
            query_pos_enc = self.array_adapter(raw).expand(batch_size, -1, -1)
        else:
            query_pos_enc = self.query_pos_enc_cached if query_indices is None else self.query_pos_enc_cached[:, query_indices, :]

        label_gt = query_rss
        label_gt, _ = scale_in_last_dim(label_gt)

        pred_rss, _ = estimator(sample_rss, sample_pos_enc, query_pos_enc, grid_indices)

        loss, as_loss_db = self._loss_fun(label_gt, pred_rss)
        return loss, as_loss_db


    def _save_best_model(self, loss, epoch, batch_idx, estimator, generator):

        if not hasattr(self, "best_loss"):
            self.best_loss = float("inf")


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
                if self.has_array_adapter():
                    adapter_to_save = self.accelerator.unwrap_model(self.array_adapter)
                    self.save_model(adapter_to_save, "best_array_adapter")

                record_path = os.path.join(self.config.training.model_save_path, "best_record.txt")
                with open(record_path, "w") as f:
                    f.write(f"Best loss: {self.best_loss:.6f} at Epoch {epoch+1}, Batch {batch_idx}\n")

    

    def _loss_fun(self, tensor_gt, tensor_pred):

        mse = torch.nn.MSELoss()

        mse_loss_as = mse(tensor_gt, tensor_pred)
        mse_loss_as_db = 10 * torch.log10(mse_loss_as + 1e-16)

        return mse_loss_as, mse_loss_as_db
    
    def train(self):
        ckpt_path = getattr(self.config.training, 'ckpt_path', None)

        # Pre-load checkpoint to sync total_epochs before scheduler creation
        ckpt = None
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            if 'total_epochs' in ckpt:
                self.config.training.epochs = ckpt['total_epochs']
                self.log(f"Synced total_epochs={ckpt['total_epochs']} from checkpoint")

        if not ckpt_path and self._check_model_exist(self.config.training.model_save_path):
            raise FileExistsError("You Use An Existing Folder as Save Path")

        generator = self.initialize_generator()
        estimator = self.initialize_estimator()
        if self.has_array_adapter():
            self.array_adapter = self.initialize_array_adapter()
        else:
            self.array_adapter = None

        scheme = self.config.training.scheme
        self.log(f"Training scheme: {scheme}")

        estimator.train()


        if self.config.estimator.optimizer.lower() == "adamw":
            estimator_params = list(estimator.parameters())
            if self.has_array_adapter():
                estimator_params += list(self.array_adapter.parameters())
            optimizer_estimator = torch.optim.AdamW(
                estimator_params, lr=self.config.training.lr_estimator, weight_decay=0.01
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

        optimizer_generator = None
        scheduler_generator = None

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

            if self.has_array_adapter():
                train_dataloader, optimizer_generator, optimizer_estimator, generator, estimator, self.array_adapter = self.accelerator.prepare(
                    self.train_dataloader, optimizer_generator, optimizer_estimator, generator, estimator, self.array_adapter
                )
            else:
                train_dataloader, optimizer_generator, optimizer_estimator, generator, estimator = self.accelerator.prepare(
                    self.train_dataloader, optimizer_generator, optimizer_estimator, generator, estimator
                )
        else:
            if self.has_array_adapter():
                train_dataloader, estimator, optimizer_estimator, self.array_adapter = self.accelerator.prepare(
                    self.train_dataloader, estimator, optimizer_estimator, self.array_adapter
                )
            else:
                train_dataloader, estimator, optimizer_estimator = self.accelerator.prepare(
                    self.train_dataloader, estimator, optimizer_estimator
                )

        start_epoch = 0
        if ckpt is not None:
            start_epoch = self.load_checkpoint(
                ckpt, estimator, generator, optimizer_estimator, scheduler_estimator,
                optimizer_generator, scheduler_generator
            )

        dp = load_data_process(self.config, device=self.device)
        if self.has_array_adapter():
            # Cache with batch_size=1 to save memory; adapter applied per-step
            self.query_pos_enc_raw = dp.generate_query_position_encoding(
                batch_size=1
            ).to(dtype=torch.float32)
            self.log(f"Query positional encoding (raw) precomputed: shape {list(self.query_pos_enc_raw.shape)}")
        else:
            self.query_pos_enc_cached = dp.generate_query_position_encoding(
                batch_size=self.config.training.batch_size
            ).to(dtype=torch.float32)
            self.log(f"Query positional encoding precomputed: shape {list(self.query_pos_enc_cached.shape)}")

        for epoch in range(start_epoch, self.config.training.epochs):
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

                if self.accelerator.is_main_process:
                    self._save_best_model(loss, epoch, batch_idx, estimator, generator)

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
                if self.has_array_adapter():
                    adapter_to_save = self.accelerator.unwrap_model(self.array_adapter)
                    self.save_model(adapter_to_save, f"array_adapter_{epoch_tag}")
                self.save_checkpoint(
                    epoch, estimator, generator, optimizer_estimator, scheduler_estimator,
                    optimizer_generator, scheduler_generator
                )
        print("Training complete.")
    
    def _get_estimator_params_for_clipping(self, estimator):
        params = list(estimator.parameters())
        if self.has_array_adapter():
            params = params + list(self.array_adapter.parameters())
        return params

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
        self.accelerator.clip_grad_norm_(self._get_estimator_params_for_clipping(estimator), 0.5)
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
        self.accelerator.clip_grad_norm_(self._get_estimator_params_for_clipping(estimator), 0.5)
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
