import time

def profile(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"[⏱ 실행 시간] {func.__name__}: {end - start:.4f}초")
        return result
    return wrapper
