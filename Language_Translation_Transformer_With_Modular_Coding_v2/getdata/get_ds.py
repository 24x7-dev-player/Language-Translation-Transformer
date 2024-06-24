from torch.utils.data import DataLoader, random_split
from datasets import load_dataset
from maketokenizers.build_tokenizer import build_tokenizer
from getdata.bilingual_dataset import BilingualDataset
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[
                        logging.FileHandler("translation.log"),
                        logging.StreamHandler()
                    ])

def get_ds(config):
    
    # Loading the train portion of the IIT Bombay English-Hindi Parallel Corpus dataset without a limit.
    ds_raw = load_dataset('cfilt/iitb-english-hindi', split='train')
    ds_raw = ds_raw.select(range(30000))  # Selecting the first 30,000 rows
    logging.info("Loaded IIT Bombay English-Hindi Dataset Successfully!")
    
    # Building or loading tokenizer for both the source and target languages 
    tokenizer_src = build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = build_tokenizer(config, ds_raw, config['lang_tgt'])
    logging.info("Source and Target Tokenizer Built Successfully!")
    
    # Splitting the dataset for training and validation 
    train_ds_size = int(0.9 * len(ds_raw)) # 90% for training
    val_ds_size = len(ds_raw) - train_ds_size # 10% for validation
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size]) # Randomly splitting the dataset
    logging.info("Splitting dataset Successfully!")
    
    logging.info("Preprocessing with Bilingual Dataset Starting ... ")                       
    # Processing data with the BilingualDataset class
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    logging.info("Preprocessing with Bilingual Dataset Successfully!")
                                    
    # Iterating over the entire dataset and printing the maximum length found in the sentences of both the source and target languages
    max_len_src = 0
    max_len_tgt = 0
    for pair in ds_raw:
        src_ids = tokenizer_src.encode(pair['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(pair['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    logging.info(f"Max length of source sentence: {max_len_src}")
    logging.info(f"Max length of target sentence: {max_len_tgt}")
    
    # Creating dataloaders for the training and validation sets
    # Dataloaders are used to iterate over the dataset in batches during training and validation
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True) # Batch size will be defined in the config dictionary
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
    
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt # Returning the DataLoader objects and tokenizers
