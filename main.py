from utils import check_gpu
from dataloader import Dataset
from dataloaderkr import Dataset_kr
from model import Models

if __name__ == '__main__':

    # 모델 생성, 학습, 모델 저장
    model = Models(num_labels = 2)
    dataset = Dataset()
    dataset.set_dataset('train')    # train dataset load
    dataset.set_dataset('test')     # test dataset load
    train, valid, test = dataset.get_dataloader()   # train/valid/test dataloader 생성
    model.BERT()    # 모델 생성
    model.train(train, valid, epochs = 1)    # 학습
    model.test(test)    # 테스트
    model.save_model()

    # krbert 모델
    # 모델 생성, 학습, 모델 저장
    model_k = Models(num_labels = 2)
    dataset = Dataset_kr()
    dataset.set_dataset('train')    # train dataset load
    dataset.set_dataset('test')     # test dataset load
    train, valid, test = dataset.get_dataloader()   # train/valid/test dataloader 생성
    model_k.KRBERT()    # 모델 생성
    model_k.train(train, valid, epochs = 1)    # 학습
    model_k.test(test)    # 테스트
    model_k.save_model()

    # # 모델 생성, 모델 불러오기, 문장 폭력성 여부 추론
    # model = Models(num_labels = 2)
    # model.load_model()
    # model.inference('이런건 걍 돈없고 멍청한 전라도 여자들 겨냥한 상술이지') 
    # 같은 함수 명이라면 헷갈릴것 같아 파일을 나누면서 함수 이름도 살짝 고쳤습니다.
    # model_k.load_model()
    # model_k.inference('이런건 걍 돈없고 멍청한 전라도 여자들 겨냥한 상술이지') 