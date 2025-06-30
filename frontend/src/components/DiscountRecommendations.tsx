import React, { useState, useEffect } from 'react';
import { 
  AlertTriangle, 
  Clock, 
  DollarSign, 
  CheckCircle, 
  XCircle, 
  Edit3,
  TrendingDown,
  Package,
  MapPin,
  Brain,
  ThumbsUp,
  ThumbsDown,
  Database,
  Target
} from 'lucide-react';
import { DiscountRecommendationService } from '../services/discountRecommendationService';
import { UserTrainingService } from '../services/userTrainingService';

interface DiscountRecommendation {
  id: string;
  productName: string;
  category: string;
  currentStock: number;
  hoursUntilExpiry: number;
  predictedWasteAmount: number;
  predictedWastePercentage: number;
  suggestedDiscountPercentage: number;
  estimatedRevenueSaved: number;
  confidence: number;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  reasoning: string[];
  location: string;
  currentPrice: number;
  discountedPrice: number;
  status: 'pending' | 'approved' | 'rejected' | 'applied';
  timestamp: Date;
  basedOnDataPoints: number;
  historicalWasteRate: number;
}

interface DiscountRecommendationsProps {
  trainedModelData: any;
}

const DiscountRecommendations: React.FC<DiscountRecommendationsProps> = ({ trainedModelData }) => {
  const [recommendations, setRecommendations] = useState<DiscountRecommendation[]>([]);
  const [customDiscounts, setCustomDiscounts] = useState<{[key: string]: number}>({});
  const [notes, setNotes] = useState<{[key: string]: string}>({});
  const [stats, setStats] = useState<any>(null);
  const [showHistory, setShowHistory] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  
  const discountService = DiscountRecommendationService.getInstance();
  const userTrainingService = UserTrainingService.getInstance();

  useEffect(() => {
    if (trainedModelData) {
      const rawTrainingData = userTrainingService.getTrainingData();
      discountService.setTrainedModelData(trainedModelData, rawTrainingData);
      generateRecommendations();
    }
  }, [trainedModelData]);

  const generateRecommendations = async () => {
    setIsGenerating(true);
    try {
      const rawTrainingData = userTrainingService.getTrainingData();
      discountService.setTrainedModelData(trainedModelData, rawTrainingData);
      
      // Small delay to show loading state
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const newRecommendations = discountService.generateRecommendationsFromTrainingData();
      setRecommendations(discountService.getPendingRecommendations());
      setStats(discountService.getRecommendationStats());
    } catch (error) {
      console.error('Error generating recommendations:', error);
      alert('Error generating recommendations: ' + (error as Error).message);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleApprove = (recommendationId: string) => {
    const customDiscount = customDiscounts[recommendationId];
    const ownerNotes = notes[recommendationId];
    
    discountService.approveRecommendation(recommendationId, customDiscount, ownerNotes);
    setRecommendations(discountService.getPendingRecommendations());
    setStats(discountService.getRecommendationStats());
    
    // Clear custom inputs
    setCustomDiscounts(prev => ({ ...prev, [recommendationId]: 0 }));
    setNotes(prev => ({ ...prev, [recommendationId]: '' }));
  };

  const handleReject = (recommendationId: string) => {
    const ownerNotes = notes[recommendationId];
    
    discountService.rejectRecommendation(recommendationId, ownerNotes);
    setRecommendations(discountService.getPendingRecommendations());
    setStats(discountService.getRecommendationStats());
    
    // Clear notes
    setNotes(prev => ({ ...prev, [recommendationId]: '' }));
  };

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'critical': return 'bg-red-100 text-red-800 border-red-200';
      case 'high': return 'bg-orange-100 text-orange-800 border-orange-200';
      case 'medium': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      default: return 'bg-green-100 text-green-800 border-green-200';
    }
  };

  const getUrgencyIcon = (hoursUntilExpiry: number) => {
    if (hoursUntilExpiry <= 6) return <AlertTriangle className="w-4 h-4 text-red-500" />;
    if (hoursUntilExpiry <= 12) return <Clock className="w-4 h-4 text-orange-500" />;
    return <Clock className="w-4 h-4 text-yellow-500" />;
  };

  if (!trainedModelData) {
    return (
      <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
        <div className="text-center py-8">
          <Brain className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-800 mb-2">No Trained Model Available</h3>
          <p className="text-gray-600">Please train your AI model first to get discount recommendations based on your data.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Training Data Info */}
      <div className="bg-blue-50 border border-blue-200 rounded-xl p-4">
        <div className="flex items-center space-x-3">
          <Database className="w-5 h-5 text-blue-600" />
          <div>
            <h4 className="font-medium text-blue-800">
              Recommendations Based on Your Training Data
            </h4>
            <p className="text-sm text-blue-600">
              Using {trainedModelData.totalRecords} data points across {trainedModelData.categories} categories
            </p>
          </div>
        </div>
      </div>

      {/* Stats Overview */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-100">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Total Recommendations</p>
                <p className="text-2xl font-bold text-gray-800">{stats.total}</p>
              </div>
              <Brain className="w-8 h-8 text-blue-500" />
            </div>
          </div>
          
          <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-100">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Approval Rate</p>
                <p className="text-2xl font-bold text-green-600">{stats.approvalRate}%</p>
              </div>
              <ThumbsUp className="w-8 h-8 text-green-500" />
            </div>
          </div>
          
          <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-100">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Pending</p>
                <p className="text-2xl font-bold text-orange-600">{stats.pending}</p>
              </div>
              <Clock className="w-8 h-8 text-orange-500" />
            </div>
          </div>
          
          <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-100">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Revenue Saved</p>
                <p className="text-2xl font-bold text-blue-600">${stats.totalPotentialSavings}</p>
              </div>
              <DollarSign className="w-8 h-8 text-blue-500" />
            </div>
          </div>
        </div>
      )}

      {/* Header */}
      <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-xl font-bold text-gray-800">AI Discount Recommendations</h2>
            <p className="text-gray-600">Based on your actual training data patterns</p>
          </div>
          <div className="flex space-x-3">
            <button
              onClick={generateRecommendations}
              disabled={isGenerating}
              className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors duration-200 flex items-center space-x-2 disabled:opacity-50"
            >
              <Target className={`w-4 h-4 ${isGenerating ? 'animate-spin' : ''}`} />
              <span>{isGenerating ? 'Analyzing Data...' : 'Generate New Recommendations'}</span>
            </button>
            <button
              onClick={() => setShowHistory(!showHistory)}
              className="bg-gray-100 text-gray-700 px-4 py-2 rounded-lg hover:bg-gray-200 transition-colors duration-200"
            >
              {showHistory ? 'Hide History' : 'Show History'}
            </button>
          </div>
        </div>
      </div>

      {/* Loading State */}
      {isGenerating && (
        <div className="bg-white p-8 rounded-xl shadow-sm border border-gray-100 text-center">
          <div className="animate-spin w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full mx-auto mb-4"></div>
          <h3 className="text-lg font-medium text-gray-800 mb-2">Analyzing Your Training Data</h3>
          <p className="text-gray-600">
            AI is processing your historical waste patterns to generate personalized recommendations...
          </p>
        </div>
      )}

      {/* Recommendations List */}
      {!isGenerating && recommendations.length > 0 ? (
        <div className="space-y-4">
          {recommendations.map((recommendation) => (
            <div key={recommendation.id} className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-start space-x-4">
                  <div className="flex-shrink-0">
                    {getUrgencyIcon(recommendation.hoursUntilExpiry)}
                  </div>
                  <div className="flex-1">
                    <h3 className="text-lg font-semibold text-gray-800">{recommendation.productName}</h3>
                    <div className="flex items-center space-x-4 text-sm text-gray-600 mt-1">
                      <span className="flex items-center space-x-1">
                        <Package className="w-3 h-3" />
                        <span>{recommendation.currentStock} units</span>
                      </span>
                      <span className="flex items-center space-x-1">
                        <MapPin className="w-3 h-3" />
                        <span>{recommendation.location}</span>
                      </span>
                      <span className="flex items-center space-x-1">
                        <Clock className="w-3 h-3" />
                        <span>{recommendation.hoursUntilExpiry}h until expiry</span>
                      </span>
                      <span className="flex items-center space-x-1">
                        <Database className="w-3 h-3" />
                        <span>Based on {recommendation.basedOnDataPoints} data points</span>
                      </span>
                    </div>
                  </div>
                </div>
                <div className={`px-3 py-1 rounded-full text-xs font-medium border ${getRiskColor(recommendation.riskLevel)}`}>
                  {recommendation.riskLevel.toUpperCase()} RISK
                </div>
              </div>

              {/* Historical Context */}
              <div className="bg-blue-50 p-3 rounded-lg mb-4">
                <div className="flex items-center space-x-2 mb-1">
                  <Database className="w-4 h-4 text-blue-600" />
                  <span className="text-sm font-medium text-blue-800">Historical Data Analysis</span>
                </div>
                <p className="text-sm text-blue-700">
                  Historical waste rate for this product/category: <strong>{recommendation.historicalWasteRate}%</strong>
                  {' '}(based on {recommendation.basedOnDataPoints} training data points)
                </p>
              </div>

              {/* Prediction Details */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4 p-4 bg-gray-50 rounded-lg">
                <div className="text-center">
                  <p className="text-sm text-gray-600">Predicted Waste</p>
                  <p className="text-lg font-bold text-red-600">
                    {recommendation.predictedWasteAmount} units ({recommendation.predictedWastePercentage}%)
                  </p>
                </div>
                <div className="text-center">
                  <p className="text-sm text-gray-600">Suggested Discount</p>
                  <p className="text-lg font-bold text-blue-600">{recommendation.suggestedDiscountPercentage}%</p>
                </div>
                <div className="text-center">
                  <p className="text-sm text-gray-600">Estimated Revenue Saved</p>
                  <p className="text-lg font-bold text-green-600">${recommendation.estimatedRevenueSaved}</p>
                </div>
              </div>

              {/* Pricing */}
              <div className="flex items-center space-x-4 mb-4">
                <div>
                  <span className="text-sm text-gray-600">Current Price: </span>
                  <span className="text-lg font-semibold text-gray-800">${recommendation.currentPrice}</span>
                </div>
                <div>
                  <span className="text-sm text-gray-600">Discounted Price: </span>
                  <span className="text-lg font-semibold text-green-600">${recommendation.discountedPrice}</span>
                </div>
                <div className="flex items-center space-x-1">
                  <span className="text-sm text-gray-600">Confidence: </span>
                  <span className="text-sm font-medium text-blue-600">{recommendation.confidence}%</span>
                </div>
              </div>

              {/* AI Reasoning */}
              <div className="mb-4">
                <h4 className="text-sm font-medium text-gray-700 mb-2">AI Reasoning (Based on Your Data):</h4>
                <ul className="text-sm text-gray-600 space-y-1">
                  {recommendation.reasoning.map((reason, index) => (
                    <li key={index} className="flex items-start space-x-2">
                      <span className="text-blue-500 mt-1">•</span>
                      <span>{reason}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Custom Discount Input */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Custom Discount % (Optional)
                  </label>
                  <input
                    type="number"
                    min="0"
                    max="70"
                    value={customDiscounts[recommendation.id] || ''}
                    onChange={(e) => setCustomDiscounts(prev => ({
                      ...prev,
                      [recommendation.id]: parseInt(e.target.value) || 0
                    }))}
                    className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    placeholder={`AI Suggested: ${recommendation.suggestedDiscountPercentage}%`}
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Notes (Optional)
                  </label>
                  <input
                    type="text"
                    value={notes[recommendation.id] || ''}
                    onChange={(e) => setNotes(prev => ({
                      ...prev,
                      [recommendation.id]: e.target.value
                    }))}
                    className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    placeholder="Add your notes..."
                  />
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex items-center justify-end space-x-3">
                <button
                  onClick={() => handleReject(recommendation.id)}
                  className="bg-red-100 text-red-700 px-4 py-2 rounded-lg hover:bg-red-200 transition-colors duration-200 flex items-center space-x-2"
                >
                  <XCircle className="w-4 h-4" />
                  <span>Reject</span>
                </button>
                <button
                  onClick={() => handleApprove(recommendation.id)}
                  className="bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 transition-colors duration-200 flex items-center space-x-2"
                >
                  <CheckCircle className="w-4 h-4" />
                  <span>Approve Discount</span>
                </button>
              </div>
            </div>
          ))}
        </div>
      ) : !isGenerating ? (
        <div className="bg-white p-8 rounded-xl shadow-sm border border-gray-100 text-center">
          <TrendingDown className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-800 mb-2">No Discount Recommendations</h3>
          <p className="text-gray-600 mb-4">
            Based on your training data, no products currently show significant waste risk requiring discounts.
          </p>
          <button
            onClick={generateRecommendations}
            className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors duration-200"
          >
            Analyze Again
          </button>
        </div>
      ) : null}

      {/* Decision History */}
      {showHistory && (
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Decision History</h3>
          <div className="space-y-3">
            {discountService.getDecisionHistory().slice(0, 10).map((decision, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  {decision.decision === 'approved' ? (
                    <CheckCircle className="w-5 h-5 text-green-500" />
                  ) : (
                    <XCircle className="w-5 h-5 text-red-500" />
                  )}
                  <div>
                    <p className="font-medium text-gray-800">
                      {decision.decision === 'approved' ? 'Approved' : 'Rejected'} Recommendation
                      {decision.actualDiscountApplied && ` (${decision.actualDiscountApplied}% discount)`}
                    </p>
                    {decision.ownerNotes && (
                      <p className="text-sm text-gray-600">"{decision.ownerNotes}"</p>
                    )}
                  </div>
                </div>
                <span className="text-sm text-gray-500">
                  {decision.timestamp.toLocaleDateString()}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default DiscountRecommendations;