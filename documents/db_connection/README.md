# PostgreSQL DB에 연결하기

[optuna 공식 document - DB 연결방법](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/001_rdb.html#rdb)  
[sqlalchemy 공식 document- PostgreSQL 연결방법](https://docs.sqlalchemy.org/en/14/core/engines.html#database-urls)  
[sqlalchemy module import 오류 해결](https://mysmr01.tistory.com/10)  

- `optuna`만 설치해서는 `optuna`의 DB 호출 메소드를 활용할 수 없습니다.
- `optuna`는 dependency가 있는 라이브러리가 아주 많습니다. ㅜ _ㅜ

## 설치해야하는 모듈
### sqlalchemy
```
pip install sqlalchemy
```  

설치 후 다음 코드가 오류 없이 실행되는지 확인
```python
import sqlalchemy
```

### psycopg2
```
pip install psycopg2-binary
```

설치 후 다음 코드가 오류 없이 실행되는지 확인
```python
from sqlalchemy.engine import create_engine
```

## DB 연동
```python
import logging
import sys

import optuna

host="" # 서버 host
port=5432 # postgre default 포트
dbname="" # 직접 만들어야함
user="" # 직접 지정
password="" # 직접 지정

# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "이름"  # Unique identifier of the study.
storage_name = f"postgresql://{user}:{password}@{host}/{dbname}"
study = optuna.create_study(study_name=study_name, storage=storage_name)
```

필요 정보는 Slack을 참고하여 채워주시고, **study_name**도 추가로 임의로 입력해주세요!  
study_name은 추후 이어서 학습할 상황이 생길 경우 활용됩니다. (`load_if_exists=True`)

위 연동코드가 오류 없이 실행된다면 아래 코드도 오류 없이 실행될 것입니다.

```python
def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_float("y", -5, 16)
    return (x - 2) ** 2 + (y - 1) ** 3

study.optimize(objective, n_trials=3)
```

이전 학습에 이어서 학습을 하고 싶을 경우, 아래와 같이 `load_if_exists=True`로 주고 `study.optimize()` 메소드를 실행합니다.  

```python
study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
study.optimize(objective, n_trials=3)
```

학습 결과를 `pd.DataFrame`으로 가져와 출력합니다. 역시 `study_name` 기준 해당 스터디에 있는 모든 optimize 결과를 출력합니다.  

```python
study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
print(df)
```

해당 DB의 study_name에 해당하는 trial 중 Best trial을 확인합니다.

```python
print("Best params: ", study.best_params)
print("Best value: ", study.best_value)
print("Best Trial: ", study.best_trial)
print("Trials: ", study.trials)
```

마지막 출력 예시
```
Best params:  {'x': -1.3497126534053052, 'y': -1.277526084289553}
Best value:  -0.593237776758988
Best Trial:  FrozenTrial(number=3, values=[-0.593237776758988], datetime_start=...
```

DB에서 직접 결과가 어떻게 저장되었는지 확인하고 싶으면, pgAdmin4 등 GUI가 있는 SQL IDE를 활용하거나 ubuntu에 postgreSQL을 설치하여 SQL 언어를 통해 `SELECT` 해와서 확인하면 됩니다.  
