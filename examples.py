from lib.runners.asr_probing import *
from lib.runners.nlp_probing import *

def main():
    import numpy as np
    ##simple ASR example
    SimpleASRPipeline()(model2probe = Wav2Vec2Prober, morph_call = "common_voice", model_path = None,
                        save_checkpoints = False, dataset_language = ["ru"], dataset_name = None,
                        preprocessing_fn=prepare_probing_task_timit_2,
                       features = ['sex'], layers = list(np.arange(1, 3, 1)),  dataset_split= "test",
                       tokenizer= Wav2Vec2OProcessor, data_path= "SP", device = torch.device('cuda'), data_column = "data")
    
    ##simple nlp examples

    SimpleNLPPipeline(model2probe = T5Prober, dataset_name = "t5", ##important!!!!
                       model_path = None,
                       dataset_type = "person",  save_checkpoints = True, download_data = True,
                       checkpoint_path = torch.load("t5small.pth").state_dict(),
                       feature = 'label', layers = list(np.arange(1, 5, 1)), 
                       tokenizer= T5Processor, data_path= "person.tsv", device = torch.device('cuda'), data_column = "text")


    SimpleNLPPipeline(model2probe = T5EncoderDecoderProber, dataset_name = "t5",
                      model_path = None,
                      dataset_type = "person",  save_checkpoints = False, download_data = False,
                      checkpoint_path = torch.load("t5small.pth").state_dict(),
                      feature = 'label', layers = {"encoder": list(np.arange(1, 5, 1)), "decoder": list(np.arange(1, 5, 1))}, 
                      tokenizer= T5Processor, data_path= "person.tsv", device = torch.device('cuda'), data_column = "text")




    SimpleNLPPipeline(model2probe = BertOProber, dataset_name = "lovethisdataset2", model_path = 'bert-base-cased',
                       morph_call = "DiscoEval",  save_checkpoints = False, 
                       feature = 'label', layers = list(np.arange(1, 3, 1)),  dataset_split= "all",
                       tokenizer= BertProcessor, data_path= "SP", device = torch.device('cuda'), data_column = "data")
    
    SimpleNLPPipeline(model2probe = BertOProber, dataset_name = "lovethisdataset2", model_path = 'bert-base-cased',
                       checkpoint = "bert-base-cased",
                       morph_call = "DiscoEval",  save_checkpoints = False, 
                       feature = 'label', layers = list(np.arange(1, 3, 1)),  dataset_split= "dev;test", download_data = True,
                       tokenizer= BertProcessor, data_path= "DC", device = torch.device('cuda'), data_column = "data")

    layers = list(np.arange(1, 2, 1))
    SimpleNLPPipeline(model2probe = BertOProber, dataset_name = "bert", model_path = None,
                       dataset_type = "senteval",  save_checkpoints = False, dataset_split = 'train',
                       feature = 'label', layers = list(np.arange(1, 2, 1)), 
                       tokenizer= BertProcessor, data_path= "past_present.txt", device = torch.device('cuda'), data_column = "data")


if __name__ == "__main__": main()
