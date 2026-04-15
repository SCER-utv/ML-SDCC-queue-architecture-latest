class PayloadBuilder:
    """Costruisce e formatta i path S3 per il payload del Client."""

    def __init__(self, bucket, metadata):
        self.bucket = bucket
        self.metadata = metadata

    def build_paths(self, mode, dataset_info):
        """Metodo principale che smista la logica."""
        if dataset_info['is_custom']:
            return self._build_custom_paths(mode, dataset_info)
        return self._build_discovery_paths(mode, dataset_info)

    def _build_custom_paths(self, mode, dataset_info):
        """Gestisce esclusivamente la logica dei dataset forniti dall'utente."""
        train_url = dataset_info.get('train_url')
        test_url = dataset_info.get('test_url')
        s3_key = ""

        if mode == 'train':
            pass  # train_url è già corretto

        elif mode == 'bulk_infer':
            if test_url:
                s3_key = test_url.replace(f"s3://{self.bucket}/", "")

        elif mode == 'train_and_infer':
            if dataset_info['needs_split'] and train_url:
                orig_name = train_url.split('/')[-1].replace('.csv', '')
                test_url = f"s3://{self.bucket}/splits/{orig_name}_test.csv"

        elif mode == 'infer':
            base_url = train_url or test_url or ""
            s3_key = base_url.replace(f"s3://{self.bucket}/", "")

        return train_url, test_url, s3_key

    def _build_discovery_paths(self, mode, dataset_info):
        """Gestisce esclusivamente la logica dei dataset standard da configurazione."""
        ds_name = dataset_info['name']
        ds_var = dataset_info['variant']
        meta = self.metadata[ds_name][ds_var]

        train_url, test_url, s3_key = None, None, ""

        if mode == 'train':
            train_url = f"s3://{self.bucket}/{meta['train_path']}"

        elif mode == 'bulk_infer':
            test_url = f"s3://{self.bucket}/{meta['test_path']}"
            s3_key = meta['test_path']

        elif mode == 'train_and_infer':
            if dataset_info['needs_split']:
                train_url = f"s3://{self.bucket}/{meta['interim_path']}"
                orig_name = meta['interim_path'].split('/')[-1].replace('.csv', '')
                test_url = f"s3://{self.bucket}/splits/{orig_name}_test.csv"
            else:
                train_url = f"s3://{self.bucket}/{meta['train_path']}"
                test_url = f"s3://{self.bucket}/{meta['test_path']}"

        elif mode == 'infer':
            s3_key = meta['train_path']

        return train_url, test_url, s3_key