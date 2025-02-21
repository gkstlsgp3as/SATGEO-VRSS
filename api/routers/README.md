# FastAPI Query와 Alias: 간단한 가이드

## Query란

FastAPI에서 **Query**는 **`?` 뒤에 오는 URL 정보**를 가져오는 데 사용되어 **"무엇을 찾을지"** 정할 수 있음

### Query Parameter 예시

```
http://example.com/items?max-hour=10
```
- `max-hour`: **파라미터 이름**
- `10`: **값**

## FastAPI에서 Query 사용 방법

주소에 있는 정보를 코드에서 사용하려면, FastAPI의 `Query`를 사용하면 됩니다.

### 예제 1: 기본적인 Query 사용

```python
from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/items/")
def read_items(max_hour: int = Query(...)):
    return {"max_hour": max_hour}
```

#### 설명:
- `max_hour: int`는 파라미터가 **정수형이어야 함**을 의미
- `Query(...)`는 **필수 값**을 의미; 값이 없으면 FastAPI는 에러를 반환

#### 요청:
```
GET /items/?max_hour=10
```
#### 응답:
```json
{
  "max_hour": 10
}
```

---

### 예제 2: Alias로 이름 변경하기

만약 클라이언트가 `max-hour`라는 이름을 보내고, 코드에서는 `max_hr`이라는 이름을 사용하고 싶다면 **alias**를 사용할 수 있음

```python
@app.get("/items/")
def read_items(max_hr: int = Query(..., alias="max-hour")):
    return {"max_hr": max_hr}
```

#### 요청:
```
GET /items/?max-hour=10
```
#### 응답:
```json
{
  "max_hr": 10
}
```

---

## Query의 특별한 기능

### 1. 기본값 설정
클라이언트가 값을 보내지 않을 경우 기본값을 설정할 수 있음

```python
@app.get("/items/")
def read_items(max_hour: int = Query(5)):
    return {"max_hour": max_hour}
```

#### 요청:
```
GET /items/
```
#### 응답:
```json
{
  "max_hour": 5
}
```

---

### 2. 최소값 및 최대값 설정
값의 범위를 제한할 수 있음

```python
@app.get("/items/")
def read_items(max_hour: int = Query(..., ge=1, le=24)):
    return {"max_hour": max_hour}
```

- `ge=1`: 최소값은 1
- `le=24`: 최대값은 24를 초과할 수 없음

#### 요청:
```
GET /items/?max_hour=30
```
#### 응답:
```json
{
  "detail": [
    {
      "loc": ["query", "max_hour"],
      "msg": "ensure this value is less than or equal to 24",
      "type": "value_error.number.not_le",
      "ctx": {"limit_value": 24}
    }
  ]
}
```

---

## 요약

### Query란?
Query는 FastAPI에서 **URL 정보**를 가져오는 방법: 주소의 `?` 뒤에 오는 값을 처리할 수 있음

### Alias를 사용하는 경우
- 코드에서 클라이언트가 사용하는 이름과 다른 이름을 사용하고 싶을 때
- 예: 클라이언트는 `max-hour`를 보내고, 코드에서는 `max_hr`를 사용

### Query의 주요 기능
1. **필수 파라미터**: `Query(...)`에 `...`를 사용하면 필수로 설정
2. **기본값 설정**: 값이 없을 경우 기본값 지정
3. **값 범위 제한**: `ge`(최소값), `le`(최대값)를 사용해 값의 범위를 제한

---
