import os

print(' => Running necessary setup process...')
os.system('mkdir -p data/raw/')
os.system('mkdir -p data/processed/')
os.system('mkdir -p data/meta_data/')
os.system('mkdir -p data/checkpoints/')
