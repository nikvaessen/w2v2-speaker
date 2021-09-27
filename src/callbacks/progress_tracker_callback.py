################################################################################
#
# :)
#
# Author(s): Nik Vaessen
################################################################################

from pytorch_lightning import Callback, LightningModule

################################################################################
# custom callbacks for speaker identification training


class SpeakerIdentificationProgressTrackerCallback(Callback):
    """
    Callback which tracks the output of the network on a small batch
    of data.
    """

    def __init__(self, dm: VoxCelebDataModule, num_samples: int = 10):
        self.dm = dm
        self.num_samples = num_samples

        self.train_samples = self._extract_train_val_samples(dm.train_dataloader())
        self.val_samples = self._extract_train_val_samples(dm.val_dataloader())
        self.test_samples = self._extract_test_samples(dm.test_dataloader())

        print(f"tracking {len(self.train_samples)} train samples with debug output")
        print(f"tracking {len(self.val_samples)} val samples with debug output")

    @staticmethod
    def _extract_test_samples(data_loader: DataLoader):
        samples = []

        for batch in data_loader:
            (
                gt_label_batch,
                (wav_batch, other_wav_batch),
                (embedding_batch, other_embedding_batch),
            ) = batch

            for sample_idx in range(0, gt_label_batch.shape[0]):
                gt_label = gt_label_batch[sample_idx].detach()

                wav = wav_batch[sample_idx].detach()
                other_wav = other_wav_batch[sample_idx].detach()

                embedding = embedding_batch[sample_idx].detach()
                other_embedding = other_embedding_batch[sample_idx].detach()

                samples.append(
                    (gt_label, (wav, other_wav), (embedding, other_embedding))
                )

                if len(samples) >= 10:
                    return samples

        return samples

    @staticmethod
    def _extract_train_val_samples(data_loader: DataLoader):
        speaker_idxes = [0, 1, 2, 3, 4]
        samples_map = {0: [], 1: [], 2: [], 3: [], 4: []}

        for batch in data_loader:
            speaker_idx_batch, wav_batch, optional_embedding_batch = batch

            for sample_idx in range(0, speaker_idx_batch.shape[0]):
                speaker_idx = speaker_idx_batch[sample_idx].detach()
                speaker_idx_int = speaker_idx.numpy().item()

                if speaker_idx_int not in speaker_idxes:
                    continue
                if len(samples_map[speaker_idx_int]) >= 2:
                    continue

                wav = wav_batch[sample_idx].detach()
                optional_embedding = optional_embedding_batch[sample_idx].detach()

                samples_map[speaker_idx_int].append(
                    (speaker_idx, wav, optional_embedding)
                )

                if sum(len(v) for v in samples_map.values()) >= 10:
                    return [
                        tup
                        for tup in chain(
                            samples_map[0],
                            samples_map[1],
                            samples_map[2],
                            samples_map[3],
                            samples_map[4],
                        )
                    ]

        all_lists = [lst for lst in samples_map.values() if len(lst) > 0]
        all_tuples = chain(*all_lists)

        return [tup for tup in all_tuples]

    def _train_val_loop(
        self,
        name: str,
        samples: List,
        pl_module: LightningModule,
        save_dir: pathlib.Path,
        embeddings_viz_dir: pathlib.Path,
    ):
        with t.no_grad():
            embeddings = []

            for i, sample in enumerate(samples):
                speaker_idx, wav, optional_embedding = sample

                transformed_embedding = self._examine_train_val_sample(
                    pl_module,
                    speaker_idx,
                    wav,
                    optional_embedding,
                    save_dir / f"{name}_{i}.txt",
                )
                embeddings.append(transformed_embedding.squeeze().cpu())

            self._make_heatmap(embeddings, embeddings_viz_dir, name)

    def _test_loop(
        self,
        name: str,
        samples: List,
        pl_module: LightningModule,
        save_dir: pathlib.Path,
        embeddings_viz_dir: pathlib.Path,
    ):
        with t.no_grad():
            embeddings = []

            for i, sample in enumerate(samples):
                gt_label, (wav, other_wav), (embedding, other_embedding) = sample

                pp_embedding, other_pp_embedding = self._examine_test_sample(
                    pl_module,
                    gt_label,
                    wav,
                    embedding,
                    other_wav,
                    other_embedding,
                    save_dir / f"{name}_{i}.txt",
                )

                embeddings.append(pp_embedding.squeeze().cpu())
                embeddings.append(other_pp_embedding.squeeze().cpu())

            self._make_heatmap(embeddings, embeddings_viz_dir, name)

    @staticmethod
    def _make_heatmap(embeddings: List[t.Tensor], save_dir: pathlib.Path, name: str):
        heatmap_data = t.stack(embeddings)
        ax = sns.heatmap(t.stack(embeddings), robust=True)
        for i in range(heatmap_data.shape[0] + 1):
            ax.axhline(i, color="white", lw=2)

        plt.savefig(str(save_dir / f"{name}_embeddings.png"), dpi=600)
        plt.clf()

    def on_train_epoch_end(
        self, trainer, pl_module: LightningModule, outputs: Any
    ) -> None:
        save_dir = (
            extract_save_dir_path(trainer)
            / "progress"
            / f"epoch_{trainer.current_epoch}"
        )
        embeddings_viz_dir = save_dir / "embeddings"

        save_dir.mkdir(parents=True, exist_ok=True)
        embeddings_viz_dir.mkdir(parents=True, exist_ok=True)

        self._train_val_loop(
            "train", self.train_samples, pl_module, save_dir, embeddings_viz_dir
        )
        self._train_val_loop(
            "val", self.val_samples, pl_module, save_dir, embeddings_viz_dir
        )

    def on_test_end(self, trainer, pl_module: LightningModule) -> None:
        save_dir = (
            extract_save_dir_path(trainer)
            / "progress"
            / f"epoch_{trainer.current_epoch}"
        )
        embeddings_viz_dir = save_dir / "embeddings"

        save_dir.mkdir(parents=True, exist_ok=True)
        embeddings_viz_dir.mkdir(parents=True, exist_ok=True)

        self._test_loop(
            "test", self.test_samples, pl_module, save_dir, embeddings_viz_dir
        )

    @staticmethod
    def _examine_train_val_sample(
        module: LightningModule,
        speaker_idx: t.Tensor,
        wav: t.Tensor,
        optional_embedding: t.Tensor,
        output_file: pathlib.Path,
    ):
        # recreate as batch size 1
        speaker_idx = t.stack([speaker_idx]).to(module.device)
        wav = t.stack([wav]).to(module.device)
        optional_embedding = t.stack([optional_embedding]).to(module.device)

        # do the computation

        # the embedding might not have been precomputed
        if is_nan_tensor(optional_embedding):
            embedding = module.encoder_model(wav)
        else:
            embedding = optional_embedding

        transformed_embedding, predicted_speaker_idx = module(wav, optional_embedding)
        loss = module.loss_fn(predicted_speaker_idx, speaker_idx)
        accuracy = Accuracy().to(module.device)(
            t.softmax(predicted_speaker_idx, dim=1), speaker_idx
        )

        # print tensors
        with output_file.open("w") as f:
            debug_tensor_content(speaker_idx, "speaker_idx", f)
            debug_tensor_content(wav, "wav", f)
            debug_tensor_content(optional_embedding, "optional_embedding", f)

            debug_tensor_content(embedding, "embedding", f)
            debug_tensor_content(transformed_embedding, "transformed_embedding", f)
            debug_tensor_content(predicted_speaker_idx, "predicted_speaker_idx", f)
            debug_tensor_content(loss, "loss", f)
            debug_tensor_content(accuracy, "accuracy", f)

            return transformed_embedding

    @staticmethod
    def _examine_test_sample(
        module: LightningModule,
        gt_label: t.Tensor,
        wav: t.Tensor,
        optional_embedding: t.Tensor,
        other_wav: t.Tensor,
        other_optional_embedding: t.Tensor,
        output_file: pathlib.Path,
    ):
        # recreate as batch size 1
        gt_label = t.stack([gt_label]).to(module.device)

        wav = t.stack([wav]).to(module.device)
        optional_embedding = t.stack([optional_embedding]).to(module.device)

        other_wav = t.stack([other_wav]).to(module.device)
        other_optional_embedding = t.stack([other_optional_embedding]).to(module.device)

        # do the computation

        # the embedding might not have been precomputed
        if is_nan_tensor(optional_embedding):
            embedding = module.encoder_model(wav)
            other_embedding = module.encoder_model(other_wav)
        else:
            embedding = optional_embedding
            other_embedding = other_optional_embedding

        transformed_embedding, _ = module(wav, embedding)
        other_transformed_embedding, _ = module(other_wav, other_embedding)

        pp_embedding = module.post_processor_model(transformed_embedding)
        other_pp_embedding = module.post_processor_model(other_transformed_embedding)

        prediction = module.similarity_model(
            transformed_embedding, other_transformed_embedding
        )

        with output_file.open("w") as f:
            debug_tensor_content(gt_label, "gt_label", f)
            debug_tensor_content(wav, "wav", f)
            debug_tensor_content(optional_embedding, "optional_embedding", f)
            debug_tensor_content(transformed_embedding, "transformed_embedding", f)
            debug_tensor_content(pp_embedding, "post_processed_embedding", f)
            debug_tensor_content(other_wav, "other_wav", f)
            debug_tensor_content(
                other_optional_embedding, "other_optional_embedding", f
            )
            debug_tensor_content(
                other_transformed_embedding, "other_transformed_embedding", f
            )
            debug_tensor_content(
                other_pp_embedding, "other_post_processed_embedding", f
            )
            debug_tensor_content(prediction, "prediction", f)

        return pp_embedding, other_pp_embedding
