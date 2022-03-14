from lib.runners.asr_probing import *
from lib.runners.nlp_probing import *

def main():
    import numpy as np
    ##simple ASR example
    # SimpleASRPipeline()(model2probe = Wav2Vec2Prober, morph_call = "common_voice", model_path = None,
    #                     save_checkpoints = False, dataset_language = ["ru"], dataset_name = None,
    #                     preprocessing_fn=prepare_probing_task_,
    #                    features = ['gender'], layers = list(np.arange(1, 3, 1)),  dataset_split= "test",
    #                    tokenizer= Wav2Vec2OProcessor, data_path= "SP", device = torch.device('cuda'), data_column = "data", revision = "1.10.0")
    
    ##simple nlp examples

    # SimpleNLPPipeline(model2probe = T5Prober, dataset_name = "t5", ##important!!!!
    #                    model_path = None,
    #                    dataset_type = "person",  save_checkpoints = True, download_data = True,
    #                    checkpoint_path = torch.load("t5small.pth").state_dict(),
    #                    feature = 'label', layers = list(np.arange(1, 5, 1)), 
    #                    tokenizer= T5Processor, data_path= "person.tsv", device = torch.device('cuda'), data_column = "text")


    # SimpleNLPPipeline(model2probe = T5EncoderDecoderProber, dataset_name = "t5",
    #                   model_path = None,
    #                   dataset_type = "person",  save_checkpoints = False, download_data = False,
    #                 #    checkpoint_path = torch.load("t5small.pth").state_dict(),
    #                   feature = 'label', layers = {"encoder": list(np.arange(1, 5, 1)), "decoder": list(np.arange(1, 5, 1))}, 
    #                   tokenizer= T5Processor, data_path= "person.tsv", device = torch.device('cuda'), data_column = "text")




    # SimpleNLPPipeline(model2probe = BertOProber, dataset_name = "lovethisdataset2", model_path = 'bert-base-cased',
    #                    morph_call = "DiscoEval",  save_checkpoints = False, 
    #                    feature = 'label', layers = list(np.arange(1, 3, 1)),  dataset_split= "all",
    #                    tokenizer= BertProcessor, data_path= "SP", device = torch.device('cuda'), data_column = "data")
    
    # SimpleNLPPipeline(model2probe = BertOProber, dataset_name = "lovethisdataset2", model_path = 'bert-base-cased',
    #                    checkpoint = "bert-base-cased",
    #                    morph_call = "DiscoEval",  save_checkpoints = False, 
    #                    feature = 'label', layers = list(np.arange(1, 3, 1)),  dataset_split= "dev;test", download_data = True,
    #                    tokenizer= BertProcessor, data_path= "DC", device = torch.device('cuda'), data_column = "data")

    layers = list(np.arange(1, 25, 1))
    SimpleNLPPipeline(model2probe = BertOProber, dataset_name = "top_constituents", model_path = "bert-large-cased",
                       morph_call = "senteval",  save_checkpoints = False, dataset_split = "all", download_data=True, is_prepared=False,
                       feature = 'label', layers = layers, 
                       tokenizer= BertProcessor, data_path= "top_constituents.txt", device = torch.device('cuda'), data_column = "data")


if __name__ == "__main__": main()
