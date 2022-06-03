# 환경 설치 방법
## 종속성 라이브러리 설치
1. 파이선과 pip이 실행 가능한 환경 구축
2. 아래의 pip 명령어를 통해 KoBERT 다운로드
```
pip install git+https://git@github.com/SKTBrain/KoBERT.git@master
```
3. 공유된 구글 드라이브의 Capstone_NLP/Shin/FNDetectorParameters.pt 파일을 Parameter 폴더를 만들어 저장
## 주의사항
- SKT의 KoBERT의 라이브러리들이 Linux 환경에서는 Git 리포지토리에 저장된 setup.py나 requirements.txt을 따라서 설치하면 잘 실행되는 것으로 보임
- 그러나, Windows 환경에서는 Library간의 버전 호환 문제때문에
```!pip install git+https://git@github.com/SKTBrain/KoBERT.git@master```을 통해 설치가 불가능할 수도 있음
- 그런 경우에는 requirements.txt의 종속성 라이브러리들을 모두 직접 설치해줄 필요가 있음(아래는 requirements.txt의 라이브러리들)
```
pip install boto3
pip install gluonnlp
pip install mxnet
pip install onnxruntime
pip install sentencepiece
pip install torch
pip install transformers
```
- 위와같이 설치하면 계속 urllib3, numpy와 같은 라이브러리들이 버전이 안맞는다고 나오는데 그런 경우에는 transformers의 라이브러리에 numpy버전과 urlib3버전을 맞추고 뜨는 warning 메세지는 무시하고 실행하면 실행이 가능함
- 이후에 프로젝트 폴더에 KoBERT 리포지토리를 직접 clone하거나 파이선 라이브러리 폴더에 clone하면 이용 가능함

# 코드 설명
## FNDetectorPatcher
- 해당 파일은 SKT서버에서 BERT모델에 사용되는 파라미터와 같은 Pretrained된 설정들을 가져오는 역할을 수행함
- 초기 1회만 실행해도  .cache폴더가 자동으로 다운됨

## FNDetectionModel
- BERT모델 자체를 불러오는 class
- 실제 모델의 구현부로 inference를 호출하면 마지막 fc layer의 element wise 곱셈값과 최종 output값을 return 

## FakeNewsDetector
- FNDetectinModel을 호출해 실제 FakeNews를 판독하는 역할을 하는 최종 클래스
- Datset폴더에 있는 FNDetectorParameters.pt 파라미터를 불러와서 BERT를 초기화
- inference함수에서 does_infer_keyword를 True로 주면 infer_keyword 함수를 호출하여 실제 FakeNews를 찾는데 많은 영향을 끼친 단어를 영향력 순으로 정렬해서 pandas dataframe 형태로 return함. FakeNews인지를 나타내는 bool값 또한 return.

## FNDwithjson
- 메인 실행 파일으로 명령행 인자로 입력 파일 이름, 출력 파일 이름, 단어 길이를 주면 inputfile에 따라서 outputfile을 생성해줌
- 단어 길이는 각 단어가 가짜뉴스 탐지에 영향을 얼마나 미쳤는지를 높은순서대로 보여주는 기능에 사용되며 기입되는 숫자의 길이만큼 단어를 출력함
```
python FNDwithjson.py input_file_name.json output_file_name.json 20
```

# Patch
2022-06-01 00:37)config-agent.yaml 파일을 제외한 agent 기능까지 포함하여 commit 및 push
