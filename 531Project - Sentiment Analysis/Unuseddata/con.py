#combine all csv files in a directory.
import os
import glob
import pandas as pd
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
combined_csv.to_csv("Dec_immi.csv", index=False, encoding='utf-8-sig')
