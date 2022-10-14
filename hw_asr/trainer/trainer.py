import random
from pathlib import Path
from random import shuffle

import PIL
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from hw_asr.base import BaseTrainer
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.logger.utils import plot_spectrogram_to_buf
from hw_asr.metric.utils import calc_cer, calc_wer
from hw_asr.utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            metrics,
            optimizer,
            scaler,
            config,
            device,
            dataloaders,
            text_encoder,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
            mixed_precision=True,
            log_step=50
    ):
        super().__init__(model, criterion, metrics, optimizer,
                         scaler, config, device, mixed_precision)
        self.skip_oom = skip_oom
        self.text_encoder = text_encoder
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"}
        self.lr_scheduler = lr_scheduler
        self.log_step = log_step

        self.train_metrics = MetricTracker(
            "loss", "grad norm", *[m.name for m in self.metrics], writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            "loss", *[m.name for m in self.metrics], writer=self.writer
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["spectrogram", "text_encoded"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(
                ), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            # log first batch
            log_batch = batch_idx == 0 and epoch == self.start_epoch
            if log_batch:
                self._log_batch(batch, message="\nFirst input batch:")
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
                if log_batch:
                    msg = "First output batch:"
                    keys_to_log = ("logits", "log_probs", "log_probs_length")
                    self._log_batch(batch, msg, keys_to_log)
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning(
                        "OOM on batch. Skipping batch.\n", exc_info)
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler.get_last_lr()[0]
                )
                self._log_predictions(**batch)
                self._log_spectrogram(batch["spectrogram"])
                self._log_audio(**batch)
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name,
                       value in val_log.items()})

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        # self.logger.info(self.device, type(self.device), self.device.__repr__())
        with torch.autocast(device_type=self.device.type, dtype=self.mixed_dtype, enabled=self.mixed_precision):
            batch = self.move_batch_to_device(batch, self.device)
            if is_train:
                self.optimizer.zero_grad()
            outputs = self.model(**batch)
            if type(outputs) is dict:
                batch.update(outputs)
            else:
                batch["logits"] = outputs

            batch["log_probs"] = F.log_softmax(batch["logits"], dim=-1)
            batch["log_probs_length"] = self.model.transform_input_lengths(
                batch["spectrogram_length"]
            )
            batch["loss"] = self.criterion(**batch)
        if is_train:
            self.scaler.scale(batch["loss"]).backward()
            # unscale in-plcae for clipping
            self.scaler.unscale_(self.optimizer)
            self._clip_grad_norm()
            self.scaler.step(self.optimizer)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.scaler.update()

        metrics.update("loss", batch["loss"].item())
        for met in self.metrics:
            metrics.update(met.name, met(**batch))
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.evaluation_metrics,
                )
            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_predictions(**batch)
            self._log_spectrogram(batch["spectrogram"])
            self._log_audio(**batch)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        return self.evaluation_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_predictions(
            self,
            text,
            log_probs,
            log_probs_length,
            audio_path,
            examples_to_log=4,
            *args,
            **kwargs,
    ):
        if self.writer is None:
            return

        beam_search_hypos = [self.text_encoder.ctc_beam_search(
            log_probs[i].exp(), log_probs_length[i], 20) for i in range(examples_to_log)]
        
        beam_search_texts = [hypos[0].text for hypos in beam_search_hypos]

        argmax_inds = log_probs.cpu().argmax(-1).numpy()
        argmax_inds = [
            inds[: int(ind_len)]
            for inds, ind_len in zip(argmax_inds, log_probs_length.numpy())
        ]
        argmax_texts_raw = [self.text_encoder.decode(
            inds) for inds in argmax_inds]
        argmax_texts = [self.text_encoder.ctc_decode(
            inds) for inds in argmax_inds]
        tuples = list(zip(argmax_texts, beam_search_texts, text, argmax_texts_raw, audio_path))
        shuffle(tuples)
        rows = {}
        for pred, beam_pred, target, raw_pred, audio_path in tuples[:examples_to_log]:
            target = BaseTextEncoder.normalize_text(target)
            wer = calc_wer(target, pred) * 100
            cer = calc_cer(target, pred) * 100
            beam_wer = calc_wer(target, beam_pred) * 100
            beam_cer = calc_cer(target, beam_pred) * 100

            rows[Path(audio_path).name] = {
                "target": target,
                "raw prediction": raw_pred,
                "predictions": pred,
                "top beam search prediciton": beam_pred,
                "argmax_wer": wer,
                "argmax_cer": cer,
                "beam_search_wer": beam_wer,
                "beam_search_cer": beam_cer,
            }
        self.writer.add_table(
            "predictions", pd.DataFrame.from_dict(rows, orient="index"))

    def _log_spectrogram(self, spectrogram_batch):
        spectrogram = random.choice(spectrogram_batch.cpu())
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image("spectrogram", ToTensor()(image))
        
    def _log_audio(self, audio, audio_path, **kwargs):
        idx = random.randint(0, len(audio)-1)
        audio = audio[idx]
        path = audio_path[idx]
        self.writer.add_audio(path, audio, self.config["sr"])

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu()
                 for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(
                f"{metric_name}", metric_tracker.avg(metric_name))

    def _log_batch(self, batch,  message="Batch:", keys_to_log=None):
        self.logger.info(message)
        for (key, value) in batch.items():
            if not keys_to_log or key in keys_to_log:
                if "length" not in key and isinstance(value, torch.Tensor):
                    self.logger.info(f"\t{key}.shape: {value.shape}")
                else:
                    self.logger.info(f"\t{key}: {value}")
