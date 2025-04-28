# Real-Time Cyber Threat Detection and Response System Using Machine Learning Techniques

## Project Overview
This project focuses on building a real-time cyber threat detection and response system powered by machine learning. It detects, analyzes, and mitigates various cybersecurity threats like Distributed Denial of Service (DDoS) attacks, phishing, and malware infections with high accuracy and low latency.

## Features
- Real-time data collection and monitoring
- Advanced preprocessing and feature engineering
- Threat detection using machine learning models (Random Forest, Isolation Forest, LSTM)
- Automated threat response actions (IP blocking, alert generation)
- User-friendly dashboard for visualization and system control
- Scalable and adaptable to emerging threats

## Technologies Used
- Python
- Scikit-learn, TensorFlow
- Pandas, NumPy
- Streamlit (for Dashboard)
- Docker (for containerization - optional)

## Dataset
- NSL-KDD

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/real-time-cyber-threat-detection.git
   cd real-time-cyber-threat-detection
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage
- Capture real-time network traffic
- Predict and classify threats using trained machine learning models
- Visualize results on the dashboard
- Automatically block malicious IP addresses and send security alerts

## Project Structure
```
/data           - Sample datasets and processed data
/models         - Pre-trained machine learning models
/scripts        - Python scripts for data processing, training, and detection
/app.py         - Streamlit dashboard application
/README.md      - Project description
```

## Future Work
- Integration with global threat intelligence feeds
- Blockchain-based secure logging
- Reinforcement learning for adaptive responses
- Enhanced support for IoT and cloud-specific security

## Author
**Meenaloshini C**  
B.Tech Computer Engineering  
Karunya Institute of Technology and Sciences
