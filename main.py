from utils import check_gpu
from dataloader import Dataset
from model import Models

if __name__ == '__main__':

    # 모델 생성, 학습, 모델 저장
    model_b = Models(num_labels = 2)
    dataset = Dataset()
    dataset.set_dataset('train')    # train dataset load
    dataset.set_dataset('test')     # test dataset load
    train, valid, test = dataset.get_dataloader()   # train/valid/test dataloader 생성
    model_b.BERT()    # 모델 생성
    model_b.train(train, valid, epochs = 1)    # 학습
    model_b.test(test)    # 테스트
    model_b.save_model()

    # krbert 모델
    model_k = Models(num_labels = 2)
    dataset = Dataset()
    dataset.set_dataset('train')    # train dataset load
    dataset.set_dataset('test')     # test dataset load
    dataset.get_tokenizer(BertTokenizer.from_pretrained("snunlp/KR-BERT-char16424"))  
    train, valid, test = dataset.get_dataloader()   # train/valid/test dataloader 생성
    model_k.KRBERT()    # 모델 생성
    model_k.train(train, valid, epochs = 1)    # 학습
    model_k.test(test)    # 테스트
    model_k.save_model()

    # # 모델 생성, 모델 불러오기, 문장 폭력성 여부 추론
    # model = Models(num_labels = 2)
    # model.load_model()
    # model.inference('이런건 걍 돈없고 멍청한 전라도 여자들 겨냥한 상술이지')