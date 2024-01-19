from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def root():
    return {"hello":"world"}

@app.get('/home')
def root2():
    return {"hello":"worl2d"}