import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import torch

class EmbeddingCacheManager:
    def extract_and_cache(self, dataloader, backbone, cv_fold_idx, split):
        """
        Runs backbone ONCE. Saves to disk. 
        split = 'train' or 'val' (Strict leakage prevention)
        """
        backbone.eval()
        embeddings, labels = [], []
        
        with torch.no_grad():
            for batch in dataloader:
                # Forward pass through frozen/LoRA backbone
                emb = backbone(batch['pixels']) 
                embeddings.append(emb.cpu().numpy())
                labels.append(batch['labels'].cpu().numpy())
                
        embeddings = np.vstack(embeddings)
        labels = np.concatenate(labels)
        
        # Save to disk (Parquet is fast and columnar)
        table = pa.table({
            'embeddings': [emb.tolist() for emb in embeddings],
            'labels': labels
        })
        filepath = f"cache/fold_{cv_fold_idx}_{split}.parquet"
        
        import os
        os.makedirs("cache", exist_ok=True)
        pq.write_table(table, filepath)
        
        # GPU is now free. Clear cache.
        torch.cuda.empty_cache() 
        return filepath
