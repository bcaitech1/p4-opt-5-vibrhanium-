❗ 아래 내용 중 이유에 대한 부분은 개인적인 추측과 생각으로 틀린 부분이 있을 수 있습니다. 틀린 부분은 지적해주시면 감사하겠습니다!

### optuna를 통해 최적의 모델을 찾을 때 DB를 연동하여 사용하면 좋은 점과 가능한 이유
1. study를 이어서 할 수 있다.
  - 시도한 결과를 저장소에 저장해 두면 그 정보들을 활용해서 study(regression)를 이어서 할 수 있습니다.
  - create_study() 시 load_if_exists=True 사용
  - 즉, 어딘가 study한 결과를 저장하기만 하면 그걸 불러와서 거기서부터 study 시작(재시작) 가능한 것으로 보입니다.
  
2. 하나의 study를 한 PC에서 여러 개의 프로세스가 또는 여러 개의 컴퓨터가 함께 진행할 수 있다(분산 처리).
  - DB에 study 결과를 저장하므로써 여러 개의 프로세스가 결과를 읽고 또 저장하여 함께 study를 진행할 수 있는 것으로 보입니다.
  - DB의 외부 접속을 허용하면 (허용된) 다른 컴퓨터에서도 DB에 접근 가능하므로 (접근 가능한) 모두가 study 결과를 저장하고 읽을 수 있음으로써 여러 개의 컴퓨터로 하나의 study를 진행할 수도 있습니다.
  
궁금한 점
여러 프로세스가 각각의 study를 진행하는 것보다 하나의 study를 함께 진행하는 것 더 좋을까?

### AWS를 사용하는 경우와 서버에 postgresql을 설치하여 사용하는 경우의 차이점
  - AWS를 사용하는 경우 AWS 서버에 데이터가 저장됩니다.
  - 로컬 서버에 postgresql을 설치하여 사용하는 경우 로컬 서버에 데이터가 저장됩니다.
  
### AWS를 사용하는 경우와 로컬 서버를 사용하는 경우의 같은점
  - 둘 다 study 결과를 저장하므로 두 방법 모두 이어서 study 가능
  - 둘 다 study 결과를 DB에 저장하고 공유할 수 있으므로 하나의 study를 분산 처리 가능

### Referencd
- 임종국 멘토님의 Office hour
- [db_connection/README.md](https://github.com/bcaitech1/p4-opt-5-vibrhanium-/tree/master/documents/db_connection)
