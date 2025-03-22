# **FastAPI Template for ML Model Deployment 🚀**  

This repository provides a **FastAPI** template for deploying machine learning models as an API. It is designed for seamless integration, scalability, and ease of use, making it ideal for workshops and production-ready deployments.  

## **Features**  
✅ FastAPI framework for high-performance ML API  
✅ Simple `python main.py` command to run the server  
✅ Model loading and inference endpoints  
✅ Docker support for containerized deployment  
✅ Environment variable-based configuration  
✅ Logging and error handling  
✅ Example notebook for testing the API  

## **Getting Started**  

### **1. Clone the Repository**  
```bash  
git clone https://github.com/argyadiva/fastapi_dsi_mfm1.git
cd fastapi-ml-template  
```

### **2. Create and Activate a Virtual Environment**  
```bash  
python -m venv venv  
source venv/bin/activate  # On Windows: venv\Scripts\activate  
```

### **3. Install Dependencies**  
```bash  
pip install -r requirements.txt  
```

### **4. Run the FastAPI Server**  
```bash  
python main.py  
```
The API will be available at **http://localhost:8000** 🚀  

### **5. Test the API**  
You can test the API using the interactive Swagger UI at:  
📌 **http://localhost:8000/docs**  

Alternatively, send a test request using **cURL**:  
```bash  
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"features": [1.2, 3.4, 5.6]}'  
```

---

## **Project Structure**  
```
fastapi-ml-template/
│── app/
│   ├── main.py           # FastAPI app entry point
│   ├── models/           # Pre-trained ML models
│   ├── routes/           # API endpoints
│   ├── services/         # Business logic & utilities
│   ├── config.py         # Environment configuration
│── tests/                # Unit & integration tests
│── requirements.txt      # Required dependencies
│── Dockerfile            # Containerization setup
│── README.md             # Project documentation
```

---

## **Deploying with Docker**  
Build and run the containerized API:  
```bash  
docker build -t fastapi-ml-api .  
docker run -p 8000:8000 fastapi-ml-api  
```

---

## **Contributing**  
Contributions are welcome! Feel free to submit issues or pull requests.  

---

## **License**  
This project is licensed under the **MIT License**.  

