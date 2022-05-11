from ..models.model import ImageEmbedding


class Preprocessor:
    def __init__(self, jit_path, output_dir, meta_dict):
        print(
            f"Init preproc. Avail gpus: {ray.get_gpu_ids()}. cuda avail: {torch.cuda.is_available()}"
        )
        emb = ImageEmbedding(device="cuda:0", jit_path=jit_path)
        self.bim = BatchInferModel(emb, "cuda:0")
        self.output_dir = output_dir
        self.num_cpus = int(os.environ.get("OMP_NUM_THREADS"))
        self.meta_dict = meta_dict

    # def extract_meta(self, dataset, indices):
    def extract_meta(self, ray_dataset, pyramid_factor, part_id):
        # dataset = Subset(dataset, indices=indices)
        # txds = TxDataset(dataset, tx=preprocess)

        meta_dict = self.meta_dict

        def fix_meta(ray_tup):
            fullpath, binary = ray_tup
            p = os.path.realpath(fullpath)
            file_path, dbidx = meta_dict[p]
            return {"file_path": file_path, "dbidx": dbidx, "binary": binary}

        def full_preproc(tup):
            ray_tup = fix_meta(tup)
            try:
                image = PIL.Image.open(io.BytesIO(ray_tup["binary"]))
            except PIL.UnidentifiedImageError:
                print(f'error parsing binary {ray_tup["file_path"]}')
                ## some images are corrupted / not copied properly
                ## it is easier to handle that softly
                image = None

            del ray_tup["binary"]
            if image is None:
                return []  # empty list ok?
            else:
                ray_tup["image"] = image
                return preprocess(ray_tup, factor=pyramid_factor)

        def preproc_batch(b):
            return [full_preproc(tup) for tup in b]

        dl = ray_dataset.window(blocks_per_window=20).map_batches(
            preproc_batch, batch_size=20
        )
        res = []
        for batch in dl.iter_rows():
            batch_res = self.bim(batch)
            res.extend(batch_res)
        # dl = DataLoader(txds, num_workers=1, shuffle=False,
        #                 batch_size=1, collate_fn=iden)
        # res = []
        # for batch in dl:
        #     flat_batch = sum(batch,[])
        #     batch_res = self.bim(flat_batch)
        #     res.append(batch_res)

        merged_res = pd.concat(res, ignore_index=True)
        ofile = f"{self.output_dir}/part_{part_id:04d}.parquet"

        ### TMP: parquet does not allow half prec.
        x = merged_res
        x = x.assign(vectors=TensorArray(x["vectors"].to_numpy().astype("single")))
        x.to_parquet(ofile)
        return ofile
