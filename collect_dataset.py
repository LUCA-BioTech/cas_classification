import os
import time
import pandas as pd
from tqdm import tqdm
import multiprocessing
import multiprocessing.dummy
from Bio import Entrez, SeqIO, AlignIO, SearchIO

MAX_WORKERS = 16
CHUNK_SIZE = 16
Entrez.email = "XXX@gmail.com"


def run_down(args):
    id = args['id']
    folder = args['folder']
    try:
        seconds = 3
        handle = Entrez.efetch(db="protein", id=id, rettype="gb", retmode="text")
        records = handle.read()

        print(id)

        with open('./{}/{}.gb'.format(folder, id), 'w') as file_write:
            file_write.write(records)
    except:
        time.sleep(seconds)
    return args


def download(id_file='', folder=''):
    exists_id = []
    exits_file_id = [f.path for f in os.scandir(folder) if f.is_file() and not str(f).__contains__('.DS_Store')]
    for file in exits_file_id:
        exists_id.append(os.path.basename(file).replace('.gb', ''))

    all_id = []
    for line in open(id_file, 'r').readlines():
        id = line.replace('\n', '')
        if not exists_id.__contains__(id):
            all_id.append({'id': id, 'folder': folder})

    with multiprocessing.dummy.Pool(processes=MAX_WORKERS) as pool:
        results = tqdm(
            pool.imap_unordered(run_down, all_id, chunksize=CHUNK_SIZE),
            total=len(all_id),
        )

        for result in results:
            print(result)

    print('end')


def search_by_key(search_term=''):
    handle = Entrez.esearch(db="protein", term=search_term, retmax=934808)
    record = Entrez.read(handle)
    id_list = record["IdList"]

    with open('id.txt', 'w') as file_handle:
        file_handle.write('\n'.join(id_list))

def get_fasta(id_file,gene_bank_folder,fasta_folder=''):
    all_id = []
    for line in open(id_file, 'r').readlines():
        id = line.replace('\n', '')
        all_id.append(id)
    
    exists_gene=0
    no_exist_gene=0
    for id in all_id:
        gene_des_file='./{}/{}.gb'.format(gene_bank_folder, id)
        if os.path.exists(gene_des_file):
            for record in SeqIO.parse(gene_des_file, 'genbank'):
                for feature in record.features:
                        if feature.type == 'gene':
                            exists_gene=0
                            try:
                                for gene in feature.qualifiers['gene']:
                                    if str(gene).lower().__contains__('cas'):
                                        exists_gene=1
                                        
                                        fasta_file=os.path.join(fasta_folder,str(gene).lower(),str(record.id)+'.fasta')
                                        if not os.path.exists(os.path.dirname(fasta_file)):
                                            os.mkdir(os.path.dirname(fasta_file))
                                        with open(fasta_file,'w') as file_write:
                                            file_write.write('>{}\n'.format(str(record.id)))
                                            file_write.write('{}\n'.format(str(record.seq)))
                                        print(gene,record.id)
                                        
                            except:
                                print('no gene')
                            if exists_gene==0:
                                no_exist_gene=no_exist_gene+1
                                print(no_exist_gene,record.id)
def analyze_unique_seq(folder):
    all_seq=[]
    data=[]
    folder_list = [f.path for f in os.scandir(folder) if f.is_dir() and not str(f).__contains__('.DS_Store')]
    for folder in folder_list:
        file_list = [f.path for f in os.scandir(folder) if f.is_file() and not str(f).__contains__('.DS_Store')]
        for file in file_list:
            for record in SeqIO.parse(file,'fasta'):
                if not all_seq.__contains__(str(record.seq)):
                    all_seq.append(str(record.seq))
                    data.append([folder,str(record.id),str(record.seq)])
                    print(data.__len__())
    df=pd.DataFrame(data,columns=['type','id','seq'])
    df.to_csv('all_cas.csv')
                

# search_by_key('CRISPR-associated protein')
# download('id.txt','proteins')      
# get_fasta('id.txt','proteins','fasta')
# analyze_unique_seq('./cas')