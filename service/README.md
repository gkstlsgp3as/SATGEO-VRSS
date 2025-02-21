# SQLAlchemy 사용 가이드

## 목차
1. [주요 SQLAlchemy 명령어](#1-주요-sqlalchemy-명령어)
   - [add: 데이터 추가하기](#11-add-데이터-추가하기)
   - [add_all: 여러 데이터 추가하기](#12-add_all-여러-데이터-추가하기)
   - [query: 데이터 조회하기](#13-query-데이터-조회하기)
   - [delete: 데이터 삭제하기](#14-delete-데이터-삭제하기)
   - [update: 데이터 업데이트하기](#15-update-데이터-업데이트하기)
   - [rollback: 변경사항 취소하기](#16-rollback-변경사항-취소하기)
   - [commit: 변경사항 저장](#17-commit-변경사항-저장)
   - [close: 세션 닫기](#18-close-세션-닫기)
2. [bulk 메서드](#2-bulk-메서드)
   - [bulk_save_objects: 여러 객체 저장](#21-bulk_save_objects-여러-객체-저장)
   - [bulk_insert_mappings: 딕셔너리 기반 데이터 삽입](#22-bulk_insert_mappings-딕셔너리-기반-데이터-삽입)
   - [bulk_update_mappings: 딕셔너리 기반 데이터 업데이트](#23-bulk_update_mappings-딕셔너리-기반-데이터-업데이트)
3. [db.commit()](#3-dbcommit)
   - [db.commit() 사용 이유](#31-dbcommit-사용-이유)
   - [db.commit() 사용 예시](#32-dbcommit-사용-예시)

---

## 1. 주요 SQLAlchemy 명령어

### 1.1 `add`: 데이터 추가하기

새로운 데이터를 데이터베이스에 추가

```python
from sqlalchemy.orm import Session
from app.models import User  # 예: User 테이블

db: Session = Session()

new_user = User(name="John", email="john@example.com")
db.add(new_user)
db.commit()
```

---

### 1.2 `add_all`: 여러 데이터 추가하기

한 번에 여러 개의 데이터를 추가

```python
users = [
    User(name="Alice", email="alice@example.com"),
    User(name="Bob", email="bob@example.com"),
]

db.add_all(users)
db.commit()
```

---

### 1.3 `query`: 데이터 조회하기

#### (1) 모든 데이터 조회
```python
users = db.query(User).all()
for user in users:
    print(user.name, user.email)
```

#### (2) 조건에 맞는 데이터 조회
```python
user = db.query(User).filter(User.name == "John").first()
print(user.email)

```

#### (3) 조건 결합 (AND)
```python
users = db.query(User).filter(
    and_(User.name == "Alice", User.email.like("%@example.com"))
).all()
for user in users:
    print(user.name, user.email)
```

#### (4) 조건 결합 (OR)
```python
users = db.query(User).filter(
    or_(User.name == "Alice", User.email.like("%@example.com"))
).all()
for user in users:
    print(user.name, user.email)
```

---

### 1.4 `delete`: 데이터 삭제하기

```python
user = db.query(User).filter(User.name == "John").first()
db.delete(user)
db.commit()
```

---

### 1.5 `update`: 데이터 업데이트하기

```python
user = db.query(User).filter(User.name == "John").first()
user.email = "newjohn@example.com"
db.commit()
```

---

### 1.6 `rollback`: 변경사항 취소하기

트랜잭션 중 문제가 발생했을 때 변경사항을 복구

```python
try:
    new_user = User(name="Jane", email="jane@example.com")
    db.add(new_user)
    db.commit()
except Exception as e:
    db.rollback()
```

---

### 1.7 `close`: 세션 닫기

```python
db.close()
```

---

## 2. `bulk` 메서드

### 2.1 `bulk_save_objects`: 여러 객체 저장

여러 객체를 데이터베이스에 한꺼번에 저장합니다.

```python
users = [
    User(name="Alice", email="alice@example.com"),
    User(name="Bob", email="bob@example.com"),
]

db.bulk_save_objects(users)
db.commit()
```

---

### 2.2 `bulk_insert_mappings`: 딕셔너리 기반 데이터 삽입

```python
user_data = [
    {"name": "Charlie", "email": "charlie@example.com"},
    {"name": "Dana", "email": "dana@example.com"},
]

db.bulk_insert_mappings(User, user_data)
db.commit()
```

---

### 2.3 `bulk_update_mappings`: 딕셔너리 기반 데이터 업데이트

```python
user_updates = [
    {"id": 1, "email": "newalice@example.com"},
    {"id": 2, "email": "newbob@example.com"},
]

db.bulk_update_mappings(User, user_updates)
db.commit()
```
---

## 3. db.commit() 

`db.commit()`은 데이터베이스에 실행한 명령을 **확정**(영구적으로 저장)하는 역할
`add`, `delete`, `bulk` 메서드 등 데이터 조작 명령 이후에 사용

### 3.1 `db.commit()` 사용 이유

1. **안전한 트랜잭션 관리**
   - 트랜잭션: 데이터베이스 작업의 묶음
   - 모든 작업이 성공한 경우에만 데이터를 저장하기 위해 `commit()`을 사용하여 문제가 발생하면 `rollback()`으로 복구

2. **작업 최적화**
   - 여러 작업을 한 번의 트랜잭션으로 묶어 `commit`하면 효율적
   - 여러 `add` 명령 이후 한 번의 `commit`으로 모든 데이터를 저장할 수 있음

### 3.2. `db.commit()` 사용 예시
```python
new_user = User(name="Alice", email="alice@example.com")
db.add(new_user)
db.commit()
```

---
