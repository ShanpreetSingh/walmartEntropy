# SmartWaste AI Backend

## 🧠 Overview

The SmartWaste AI Backend is a comprehensive machine learning system designed to predict product waste and recommend corrective actions for retail inventory management. This system helps retailers reduce waste, optimize inventory, and maximize revenue through intelligent predictions and actionable recommendations.

## ✨ Key Features

### 🤖 Machine Learning Models
- **Waste Prediction Model**: Predicts probability of products expiring unsold
- **Action Recommendation Engine**: Suggests optimal actions (discount, donate, reroute, keep)
- **Dynamic Discount Calculator**: Calculates optimal discount percentages
- **Real-time Risk Assessment**: Continuous monitoring and risk evaluation

### 📊 Analytics & Insights
- **Waste Trend Analysis**: Historical waste patterns and trends
- **Performance Metrics**: Model accuracy and prediction effectiveness
- **Action Outcome Tracking**: Success rates of recommended actions
- **Risk Distribution**: Real-time inventory risk categorization

### 🔌 REST API Endpoints
- **Prediction API**: Batch waste prediction for inventory items
- **Analytics API**: Comprehensive reporting and trend analysis
- **Inventory Management**: Real-time inventory status and updates
- **Model Management**: Model retraining and performance monitoring

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd smartwaste-backend
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

The API will be available at `http://localhost:5000`

## 📡 API Endpoints

### Waste Prediction
```http
POST /api/predict/waste
Content-Type: application/json

{
  "inventory_items": [
    {
      "product_id": "PROD_001",
      "product_name": "Organic Bananas",
      "category": "produce",
      "current_stock": 45,
      "purchase_date": "2024-01-15",
      "expiry_date": "2024-01-22",
      "price": 2.99,
      "discount_rate": 0.0,
      "sales_velocity_7d": 8.5,
      "temperature": 68,
      "humidity": 65,
      "location": "Produce Section A",
      "day_of_week": 1,
      "is_weekend": false,
      "promotion_active": false
    }
  ]
}
```

**Response:**
```json
{
  "status": "success",
  "summary": {
    "total_products_analyzed": 1,
    "high_risk_products": 0,
    "total_predicted_waste_units": 8,
    "action_breakdown": {
      "discount": 1
    }
  },
  "predictions": [
    {
      "product_id": "PROD_001",
      "will_expire_unsold": 1,
      "waste_probability": 0.65,
      "predicted_waste_amount": 8,
      "confidence_score": 0.87,
      "recommended_action": "discount",
      "action_urgency": "medium",
      "suggested_discount": 25,
      "reasoning": [
        "Moderate waste probability (65.0%) warrants preventive measures",
        "Product expires in 7 days - proactive measures recommended",
        "Fresh produce requires rapid action due to short shelf life"
      ],
      "risk_factors": ["medium_waste_risk", "perishable_category"],
      "days_until_expiry": 7
    }
  ]
}
```

### Inventory Status
```http
GET /api/inventory/status
```

### Analytics & Trends
```http
GET /api/analytics/waste-trends?days=30&category=produce
```

### Action Recommendations
```http
POST /api/recommendations/actions
Content-Type: application/json

{
  "product_ids": ["PROD_001", "PROD_002"]
}
```

### Model Performance
```http
GET /api/model/performance
```

### Model Retraining
```http
POST /api/model/retrain
Content-Type: application/json

{
  "training_data": [...]  // Optional: new training data
}
```

## 🧮 Machine Learning Models

### Waste Prediction Model

The core ML model uses ensemble methods to predict waste probability:

- **Random Forest Classifier**: Handles non-linear relationships
- **Gradient Boosting**: Captures complex patterns
- **Logistic Regression**: Provides baseline predictions

**Features Used:**
- Current stock levels
- Days until expiry
- Sales velocity (7-day average)
- Environmental conditions (temperature, humidity)
- Product category
- Discount rates and promotions
- Seasonal factors
- Day of week patterns

### Action Recommendation Engine

Rule-based system enhanced with ML insights:

**Actions:**
- **Donate**: High waste risk + imminent expiry
- **Discount**: Moderate risk + time for sales acceleration
- **Reroute**: Overstocked + sufficient shelf life
- **Keep**: Low risk + normal sales patterns

**Factors Considered:**
- Waste probability threshold
- Time until expiry
- Stock-to-velocity ratio
- Category-specific rules
- Historical action effectiveness

## 📊 Data Requirements

### Input Data Format

Each inventory item should include:

**Required Fields:**
- `product_id`: Unique identifier
- `product_name`: Product name
- `category`: Product category (produce, dairy, bakery, meat, frozen)
- `current_stock`: Current inventory level
- `expiry_date`: Expiration date (YYYY-MM-DD)
- `sales_velocity_7d`: 7-day average daily sales

**Optional Fields:**
- `purchase_date`: Purchase/received date
- `price`: Current selling price
- `discount_rate`: Current discount (0.0-1.0)
- `temperature`: Storage temperature
- `humidity`: Storage humidity
- `location`: Storage location
- `is_weekend`: Weekend flag
- `promotion_active`: Promotion flag

### Supported Categories

- `produce`: Fresh fruits and vegetables
- `dairy`: Milk, cheese, yogurt products
- `bakery`: Bread, pastries, baked goods
- `meat`: Fresh meat and poultry
- `frozen`: Frozen food products
- `canned`: Canned and preserved goods
- `beverages`: Drinks and beverages
- `snacks`: Packaged snack foods

## 🔧 Configuration

### Environment Variables

Create a `.env` file:

```env
# Flask Configuration
FLASK_ENV=development
DEBUG=True
PORT=5000

# Model Configuration
MODEL_RETRAIN_INTERVAL=24  # hours
PREDICTION_CONFIDENCE_THRESHOLD=0.5

# Data Storage
DATA_DIRECTORY=./data
MODEL_DIRECTORY=./models

# Logging
LOG_LEVEL=INFO
LOG_FILE=smartwaste.log
```

### Model Parameters

Adjust model parameters in `models/waste_predictor.py`:

```python
# Model hyperparameters
RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 42,
    'class_weight': 'balanced'
}

# Risk thresholds
RISK_THRESHOLDS = {
    'low': 0.3,
    'medium': 0.5,
    'high': 0.7,
    'critical': 0.8
}
```

## 📈 Performance Monitoring

### Model Metrics

The system tracks several performance metrics:

- **Accuracy**: Overall prediction correctness
- **Precision**: True positive rate for waste predictions
- **Recall**: Coverage of actual waste events
- **F1-Score**: Balanced precision and recall
- **AUC-ROC**: Area under the ROC curve

### Action Effectiveness

Track the success of recommended actions:

- **Success Rate**: Percentage of successful interventions
- **Value Recovery**: Revenue saved through actions
- **Waste Prevention**: Amount of waste prevented
- **Response Time**: Time from prediction to action

## 🔄 Model Retraining

### Automatic Retraining

The system supports automatic model retraining:

1. **Data Collection**: Continuously collect actual outcomes
2. **Performance Monitoring**: Track prediction accuracy
3. **Trigger Conditions**: Retrain when accuracy drops
4. **Validation**: Validate new model before deployment

### Manual Retraining

Trigger manual retraining via API:

```bash
curl -X POST http://localhost:5000/api/model/retrain \
  -H "Content-Type: application/json" \
  -d '{"training_data": [...]}'
```

## 🛡️ Data Validation

### Input Validation

All API inputs are validated for:

- **Required Fields**: Ensure all mandatory fields are present
- **Data Types**: Validate correct data types
- **Value Ranges**: Check realistic value ranges
- **Date Formats**: Validate date format and logic
- **Category Values**: Ensure valid category names

### Error Handling

Comprehensive error handling with detailed messages:

```json
{
  "status": "error",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Validation failed",
    "details": {
      "validation_errors": [
        "Item 1: current_stock must be a non-negative number",
        "Item 2: expiry_date must be in YYYY-MM-DD format"
      ]
    }
  }
}
```

## 📊 Analytics Dashboard Data

### Waste Trends

```json
{
  "total_waste_amount": 245,
  "total_value_lost": 1250.75,
  "average_daily_waste": 8.17,
  "trend_direction": "decreasing",
  "category_breakdown": {
    "produce": {"amount": 120, "value": 600.25},
    "dairy": {"amount": 85, "value": 425.50}
  },
  "prediction_accuracy": {
    "overall_accuracy": 0.847,
    "correct_predictions": 156,
    "total_predictions": 184
  }
}
```

### Risk Distribution

```json
{
  "counts": {
    "low": 45,
    "medium": 23,
    "high": 12,
    "critical": 3
  },
  "percentages": {
    "low": 54.2,
    "medium": 27.7,
    "high": 14.5,
    "critical": 3.6
  },
  "total_products": 83
}
```

## 🚀 Deployment

### Production Deployment

1. **Environment Setup**
   ```bash
   export FLASK_ENV=production
   export DEBUG=False
   ```

2. **Use Production WSGI Server**
   ```bash
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

3. **Docker Deployment**
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   EXPOSE 5000
   CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
   ```

### Scaling Considerations

- **Load Balancing**: Use multiple instances behind a load balancer
- **Database**: Replace in-memory storage with persistent database
- **Caching**: Implement Redis for prediction caching
- **Monitoring**: Add application performance monitoring

## 🔍 Troubleshooting

### Common Issues

1. **Model Not Loading**
   - Check if model files exist in `models/` directory
   - Verify file permissions
   - Check logs for initialization errors

2. **Prediction Errors**
   - Validate input data format
   - Check for missing required fields
   - Verify category names are valid

3. **Performance Issues**
   - Monitor memory usage during batch predictions
   - Consider reducing batch sizes
   - Check for data validation bottlenecks

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 API Documentation

Complete API documentation is available at:
- Swagger UI: `http://localhost:5000/docs` (when implemented)
- Postman Collection: Available in `/docs` folder

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request



## 🙏 Acknowledgments

- Scikit-learn for machine learning capabilities
- Flask for the web framework
- Pandas for data manipulation
- NumPy for numerical computations

---

**Built with ❤️ for sustainable retail and waste reduction**
