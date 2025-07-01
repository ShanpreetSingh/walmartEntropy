"""
SmartWaste AI Backend - Main Flask Application
Provides REST API endpoints for waste prediction and inventory management
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from datetime import datetime, timedelta
import logging
import os
from models.waste_predictor import WastePredictor
from models.action_recommender import ActionRecommender
from services.inventory_service import InventoryService
from services.analytics_service import AnalyticsService
from utils.data_validator import DataValidator
from utils.response_formatter import ResponseFormatter

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize services
waste_predictor = WastePredictor()
action_recommender = ActionRecommender()
inventory_service = InventoryService()
analytics_service = AnalyticsService()
data_validator = DataValidator()
response_formatter = ResponseFormatter()

@app.route('/')
def index():
    """Health check endpoint"""
    return jsonify({
        "service": "SmartWaste AI Backend",
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/predict/waste', methods=['POST'])
def predict_waste():
    """
    Predict waste for inventory items
    
    Expected JSON payload:
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
    """
    try:
        # Validate request data
        if not request.json or 'inventory_items' not in request.json:
            return jsonify({
                "error": "Missing inventory_items in request body",
                "status": "error"
            }), 400

        inventory_items = request.json['inventory_items']
        
        # Validate each inventory item
        validation_errors = []
        for i, item in enumerate(inventory_items):
            errors = data_validator.validate_inventory_item(item)
            if errors:
                validation_errors.extend([f"Item {i+1}: {error}" for error in errors])
        
        if validation_errors:
            return jsonify({
                "error": "Validation failed",
                "validation_errors": validation_errors,
                "status": "error"
            }), 400

        # Generate predictions
        predictions = []
        for item in inventory_items:
            try:
                # Calculate days until expiry
                expiry_date = datetime.strptime(item['expiry_date'], '%Y-%m-%d')
                days_until_expiry = (expiry_date - datetime.now()).days
                
                # Get waste prediction
                waste_prediction = waste_predictor.predict_waste_probability(
                    current_stock=item['current_stock'],
                    days_until_expiry=days_until_expiry,
                    sales_velocity=item['sales_velocity_7d'],
                    category=item['category'],
                    temperature=item.get('temperature', 70),
                    humidity=item.get('humidity', 60),
                    discount_rate=item.get('discount_rate', 0.0),
                    is_weekend=item.get('is_weekend', False),
                    promotion_active=item.get('promotion_active', False)
                )
                
                # Get action recommendation
                action_recommendation = action_recommender.recommend_action(
                    waste_probability=waste_prediction['waste_probability'],
                    days_until_expiry=days_until_expiry,
                    current_stock=item['current_stock'],
                    sales_velocity=item['sales_velocity_7d'],
                    category=item['category']
                )
                
                prediction = {
                    "product_id": item['product_id'],
                    "will_expire_unsold": 1 if waste_prediction['waste_probability'] > 0.5 else 0,
                    "waste_probability": round(waste_prediction['waste_probability'], 3),
                    "predicted_waste_amount": waste_prediction['predicted_waste_amount'],
                    "confidence_score": waste_prediction['confidence'],
                    "recommended_action": action_recommendation['action'],
                    "action_urgency": action_recommendation['urgency'],
                    "suggested_discount": action_recommendation.get('suggested_discount', 0),
                    "reasoning": action_recommendation['reasoning'],
                    "risk_factors": waste_prediction['risk_factors'],
                    "days_until_expiry": days_until_expiry,
                    "prediction_timestamp": datetime.now().isoformat()
                }
                
                predictions.append(prediction)
                
                # Log prediction for analytics
                analytics_service.log_prediction(item['product_id'], prediction)
                
            except Exception as e:
                logger.error(f"Error predicting waste for product {item['product_id']}: {str(e)}")
                predictions.append({
                    "product_id": item['product_id'],
                    "error": f"Prediction failed: {str(e)}",
                    "will_expire_unsold": 0,
                    "recommended_action": "monitor"
                })

        # Format response
        response = response_formatter.format_prediction_response(predictions)
        
        logger.info(f"Generated predictions for {len(predictions)} products")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in predict_waste endpoint: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e),
            "status": "error"
        }), 500

@app.route('/api/inventory/status', methods=['GET'])
def get_inventory_status():
    """Get overall inventory status and risk summary"""
    try:
        # Get inventory summary
        inventory_summary = inventory_service.get_inventory_summary()
        
        # Get risk distribution
        risk_distribution = analytics_service.get_risk_distribution()
        
        # Get recent predictions
        recent_predictions = analytics_service.get_recent_predictions(limit=10)
        
        response = {
            "inventory_summary": inventory_summary,
            "risk_distribution": risk_distribution,
            "recent_predictions": recent_predictions,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
        
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in get_inventory_status: {str(e)}")
        return jsonify({
            "error": "Failed to get inventory status",
            "message": str(e),
            "status": "error"
        }), 500

@app.route('/api/analytics/waste-trends', methods=['GET'])
def get_waste_trends():
    """Get waste trends and analytics"""
    try:
        days = request.args.get('days', 30, type=int)
        category = request.args.get('category', None)
        
        trends = analytics_service.get_waste_trends(days=days, category=category)
        
        return jsonify({
            "trends": trends,
            "period_days": days,
            "category_filter": category,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        })

    except Exception as e:
        logger.error(f"Error in get_waste_trends: {str(e)}")
        return jsonify({
            "error": "Failed to get waste trends",
            "message": str(e),
            "status": "error"
        }), 500

@app.route('/api/recommendations/actions', methods=['POST'])
def get_action_recommendations():
    """Get specific action recommendations for products"""
    try:
        if not request.json or 'product_ids' not in request.json:
            return jsonify({
                "error": "Missing product_ids in request body",
                "status": "error"
            }), 400

        product_ids = request.json['product_ids']
        recommendations = []
        
        for product_id in product_ids:
            # Get current product data
            product_data = inventory_service.get_product_data(product_id)
            
            if not product_data:
                recommendations.append({
                    "product_id": product_id,
                    "error": "Product not found",
                    "recommended_action": "monitor"
                })
                continue
            
            # Get latest prediction
            latest_prediction = analytics_service.get_latest_prediction(product_id)
            
            if latest_prediction:
                recommendations.append({
                    "product_id": product_id,
                    "recommended_action": latest_prediction['recommended_action'],
                    "urgency": latest_prediction['action_urgency'],
                    "reasoning": latest_prediction['reasoning'],
                    "suggested_discount": latest_prediction.get('suggested_discount', 0),
                    "last_updated": latest_prediction['prediction_timestamp']
                })
            else:
                recommendations.append({
                    "product_id": product_id,
                    "recommended_action": "monitor",
                    "urgency": "low",
                    "reasoning": ["No recent predictions available"],
                    "suggested_discount": 0
                })
        
        return jsonify({
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        })

    except Exception as e:
        logger.error(f"Error in get_action_recommendations: {str(e)}")
        return jsonify({
            "error": "Failed to get action recommendations",
            "message": str(e),
            "status": "error"
        }), 500

@app.route('/api/model/retrain', methods=['POST'])
def retrain_model():
    """Retrain the waste prediction model with new data"""
    try:
        # Get training data from request or use historical data
        training_data = request.json.get('training_data', None)
        
        # Retrain the model
        training_result = waste_predictor.retrain_model(training_data)
        
        logger.info("Model retrained successfully")
        return jsonify({
            "message": "Model retrained successfully",
            "training_result": training_result,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        })

    except Exception as e:
        logger.error(f"Error in retrain_model: {str(e)}")
        return jsonify({
            "error": "Failed to retrain model",
            "message": str(e),
            "status": "error"
        }), 500

@app.route('/api/model/performance', methods=['GET'])
def get_model_performance():
    """Get model performance metrics"""
    try:
        performance_metrics = waste_predictor.get_performance_metrics()
        
        return jsonify({
            "performance_metrics": performance_metrics,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        })

    except Exception as e:
        logger.error(f"Error in get_model_performance: {str(e)}")
        return jsonify({
            "error": "Failed to get model performance",
            "message": str(e),
            "status": "error"
        }), 500

@app.route('/api/export/predictions', methods=['GET'])
def export_predictions():
    """Export predictions data"""
    try:
        format_type = request.args.get('format', 'json')
        days = request.args.get('days', 7, type=int)
        
        predictions_data = analytics_service.export_predictions(days=days, format_type=format_type)
        
        if format_type == 'csv':
            return predictions_data, 200, {
                'Content-Type': 'text/csv',
                'Content-Disposition': f'attachment; filename=predictions_{datetime.now().strftime("%Y%m%d")}.csv'
            }
        else:
            return jsonify({
                "predictions": predictions_data,
                "export_date": datetime.now().isoformat(),
                "period_days": days,
                "status": "success"
            })

    except Exception as e:
        logger.error(f"Error in export_predictions: {str(e)}")
        return jsonify({
            "error": "Failed to export predictions",
            "message": str(e),
            "status": "error"
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "message": "The requested endpoint does not exist",
        "status": "error"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred",
        "status": "error"
    }), 500

if __name__ == '__main__':
    # Initialize models on startup
    logger.info("Initializing SmartWaste AI models...")
    waste_predictor.initialize_model()
    action_recommender.initialize_rules()
    logger.info("SmartWaste AI Backend started successfully")
    
    # Run the application
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=os.environ.get('DEBUG', 'False').lower() == 'true'
    )