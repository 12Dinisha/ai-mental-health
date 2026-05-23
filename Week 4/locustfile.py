from locust import HttpUser, task, between
import random

class MHUser(HttpUser):
    wait_time = between(0.1, 0.5)
    
    @task(1)
    def predict_risk(self):
        features = [
            random.randint(0, 5),
            random.randint(1, 5),
            random.choice([0, 1]),
            random.randint(0, 27),
            random.randint(0, 21),
            random.choice([0, 1]),
            random.choice([0, 1]),
            random.choice([0, 1]),
            random.randint(0, 3),
            random.randint(0, 4),
            random.randint(0, 3),
            round(random.uniform(0, 1), 3),
            random.randint(-2, 2),
            random.randint(-2, 2),
        ]
        self.client.post("/predict", json={"features": features})
    
    @task(1)
    def health_check(self):
        self.client.get("/health")
