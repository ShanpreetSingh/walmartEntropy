import React, { useState, useEffect } from 'react';
import { Brain, Database, TrendingUp, AlertCircle, CheckCircle, RefreshCw, BarChart3, Download, Upload, Percent } from 'lucide-react';
import DataUploader from '../components/DataUploader';
import DiscountRecommendations from '../components/DiscountRecommendations';
import { UserTrainingService } from '../services/userTrainingService';
import { ngoPartners } from '../data/wasteTrainingData';

const DataTraining = () => {
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainingComplete, setTrainingComplete] = useState(false);
  const [dataSummary, setDataSummary] = useState<any>(null);
  const [latestSession, setLatestSession] = useState<any>(null);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);
  const [activeTab, setActiveTab] = useState<'upload' | 'recommendations'>('upload');
  
  const userTrainingService = UserTrainingService.getInstance();

  useEffect(() => {
    updateDataSummary();
    const session = userTrainingService.getLatestSession();
    if (session) {
      setLatestSession(session);
      setTrainingComplete(session.status === 'completed');
    }
  }, []);

  const updateDataSummary = () => {
    const summary = userTrainingService.getDataSummary();
    setDataSummary(summary);
  };

  const handleDataUpload = (data: any[]) => {
    const validation = userTrainingService.validateData(data);
    
    if (validation.valid) {
      userTrainingService.addTrainingData(data);
      updateDataSummary();
      setValidationErrors([]);
    } else {
      setValidationErrors(validation.errors);
    }
  };

  const handleTrainModel = async () => {
    if (!dataSummary || dataSummary.totalRecords === 0) {
      alert('Please upload training data first!');
      return;
    }

    setIsTraining(true);
    setTrainingProgress(0);
    setTrainingComplete(false);
    setValidationErrors([]);

    // Simulate training progress
    const progressInterval = setInterval(() => {
      setTrainingProgress(prev => {
        if (prev >= 90) {
          clearInterval(progressInterval);
          return 90;
        }
        return prev + Math.random() * 15;
      });
    }, 300);

    try {
      const session = await userTrainingService.trainModel();
      setLatestSession(session);
      setTrainingProgress(100);
      setTrainingComplete(true);
      
      // Auto-switch to recommendations tab after successful training
      setTimeout(() => {
        setActiveTab('recommendations');
      }, 1000);
    } catch (error) {
      console.error('Training failed:', error);
      alert('Training failed: ' + (error as Error).message);
    } finally {
      setIsTraining(false);
      clearInterval(progressInterval);
    }
  };

  const handleClearData = () => {
    if (confirm('Are you sure you want to clear all training data? This action cannot be undone.')) {
      userTrainingService.clearTrainingData();
      updateDataSummary();
      setLatestSession(null);
      setTrainingComplete(false);
      setActiveTab('upload');
    }
  };

  const handleExportData = () => {
    const csvContent = userTrainingService.exportTrainingData();
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `training_data_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  const trainingStats = [
    { 
      label: 'Training Records', 
      value: dataSummary?.totalRecords || 0, 
      icon: Database,
      color: 'blue'
    },
    { 
      label: 'Product Categories', 
      value: dataSummary?.categories || 0, 
      icon: BarChart3,
      color: 'green'
    },
    { 
      label: 'Model Accuracy', 
      value: latestSession ? `${latestSession.modelAccuracy.toFixed(1)}%` : 'Not Trained', 
      icon: TrendingUp,
      color: 'purple'
    },
    { 
      label: 'Waste Rate', 
      value: dataSummary ? `${dataSummary.wastePercentage}%` : '0%', 
      icon: AlertCircle,
      color: 'orange'
    }
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-800 mb-2">SmartWaste AI Training Center</h1>
        <p className="text-gray-600">Upload your own data to train personalized waste prediction models and get discount recommendations</p>
      </div>

      {/* Tab Navigation */}
      <div className="mb-8">
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8">
            <button
              onClick={() => setActiveTab('upload')}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'upload'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <div className="flex items-center space-x-2">
                <Upload className="w-4 h-4" />
                <span>Data Training</span>
              </div>
            </button>
            <button
              onClick={() => setActiveTab('recommendations')}
              disabled={!trainingComplete}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'recommendations' && trainingComplete
                  ? 'border-blue-500 text-blue-600'
                  : trainingComplete
                  ? 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  : 'border-transparent text-gray-400 cursor-not-allowed'
              }`}
            >
              <div className="flex items-center space-x-2">
                <Percent className="w-4 h-4" />
                <span>Discount Recommendations</span>
                {trainingComplete && (
                  <span className="bg-green-100 text-green-800 text-xs px-2 py-1 rounded-full">Ready</span>
                )}
              </div>
            </button>
          </nav>
        </div>
      </div>

      {/* Training Status */}
      <div className="bg-gradient-to-r from-blue-600 to-blue-700 rounded-2xl p-6 mb-8 text-white">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <Brain className="w-8 h-8" />
            <div>
              <h2 className="text-xl font-bold">AI Model Status</h2>
              <p className="text-blue-100">
                {dataSummary?.totalRecords > 0 
                  ? `Trained on ${dataSummary.totalRecords} data points`
                  : 'No training data uploaded yet'
                }
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            {trainingComplete ? (
              <CheckCircle className="w-6 h-6 text-green-400" />
            ) : (
              <AlertCircle className="w-6 h-6 text-yellow-400" />
            )}
            <span className="font-medium">
              {trainingComplete ? 'Model Ready' : dataSummary?.totalRecords > 0 ? 'Ready to Train' : 'Upload Data First'}
            </span>
          </div>
        </div>

        {/* Training Progress */}
        {isTraining && (
          <div className="mb-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm">Training Progress</span>
              <span className="text-sm">{Math.round(trainingProgress)}%</span>
            </div>
            <div className="w-full bg-blue-800 rounded-full h-2">
              <div 
                className="bg-gradient-to-r from-green-400 to-green-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${trainingProgress}%` }}
              ></div>
            </div>
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex flex-wrap gap-3">
          <button
            onClick={handleTrainModel}
            disabled={isTraining || !dataSummary || dataSummary.totalRecords === 0}
            className="bg-white text-blue-600 px-6 py-3 rounded-lg font-medium hover:bg-blue-50 transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
          >
            <RefreshCw className={`w-5 h-5 ${isTraining ? 'animate-spin' : ''}`} />
            <span>{isTraining ? 'Training Model...' : 'Train Model'}</span>
          </button>
          
          {dataSummary?.totalRecords > 0 && (
            <>
              <button
                onClick={handleExportData}
                className="bg-white/20 text-white px-4 py-3 rounded-lg font-medium hover:bg-white/30 transition-colors duration-200 flex items-center space-x-2"
              >
                <Download className="w-5 h-5" />
                <span>Export Data</span>
              </button>
              
              <button
                onClick={handleClearData}
                className="bg-red-500/20 text-white px-4 py-3 rounded-lg font-medium hover:bg-red-500/30 transition-colors duration-200"
              >
                Clear Data
              </button>
            </>
          )}
        </div>
      </div>

      {/* Validation Errors */}
      {validationErrors.length > 0 && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
          <h4 className="font-medium text-red-800 mb-2">Data Validation Errors:</h4>
          <ul className="text-sm text-red-700 space-y-1">
            {validationErrors.map((error, index) => (
              <li key={index}>• {error}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Training Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {trainingStats.map((stat, index) => {
          const Icon = stat.icon;
          const colorClasses = {
            blue: 'bg-blue-50 text-blue-600 border-blue-100',
            green: 'bg-green-50 text-green-600 border-green-100',
            purple: 'bg-purple-50 text-purple-600 border-purple-100',
            orange: 'bg-orange-50 text-orange-600 border-orange-100'
          };
          
          return (
            <div key={index} className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-medium text-gray-600">{stat.label}</h3>
                <div className={`p-2 rounded-lg ${colorClasses[stat.color as keyof typeof colorClasses]}`}>
                  <Icon className="w-5 h-5" />
                </div>
              </div>
              <p className="text-2xl font-bold text-gray-800">{stat.value}</p>
            </div>
          );
        })}
      </div>

      {/* Tab Content */}
      {activeTab === 'upload' ? (
        <>
          {/* Data Upload Section */}
          <div className="mb-8">
            <DataUploader onDataUpload={handleDataUpload} />
          </div>

          {/* Data Summary */}
          {dataSummary && dataSummary.totalRecords > 0 && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
              {/* Category Breakdown */}
              <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                <h3 className="text-lg font-semibold text-gray-800 mb-4">Data by Category</h3>
                <div className="space-y-3">
                  {dataSummary.categoryBreakdown.map((cat: any, index: number) => (
                    <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <div>
                        <p className="font-medium text-gray-800 capitalize">{cat.category}</p>
                        <p className="text-sm text-gray-600">{cat.count} records</p>
                      </div>
                      <div className="text-right">
                        <p className="text-sm font-medium text-red-600">{cat.waste} units wasted</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Training History */}
              <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                <h3 className="text-lg font-semibold text-gray-800 mb-4">Training Sessions</h3>
                {latestSession ? (
                  <div className="space-y-3">
                    <div className="p-3 bg-green-50 rounded-lg border border-green-200">
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-medium text-green-800">Latest Session</span>
                        <span className="text-sm text-green-600">
                          {latestSession.timestamp.toLocaleDateString()}
                        </span>
                      </div>
                      <p className="text-sm text-green-700">
                        Accuracy: {latestSession.modelAccuracy.toFixed(1)}%
                      </p>
                      <p className="text-sm text-green-700">
                        Data Points: {latestSession.dataPoints.length}
                      </p>
                    </div>
                  </div>
                ) : (
                  <p className="text-gray-500 text-center py-8">No training sessions yet</p>
                )}
              </div>
            </div>
          )}

          {/* Instructions */}
          <div className="bg-blue-50 border border-blue-200 rounded-xl p-6">
            <h3 className="text-lg font-semibold text-blue-800 mb-3">How to Train Your SmartWaste AI Model</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center">
                <div className="w-12 h-12 bg-blue-600 text-white rounded-full flex items-center justify-center mx-auto mb-3">
                  <Upload className="w-6 h-6" />
                </div>
                <h4 className="font-medium text-blue-800 mb-2">1. Upload Your Data</h4>
                <p className="text-sm text-blue-600">
                  Upload CSV files or manually enter your historical food waste data including stock levels, expiry dates, and environmental conditions
                </p>
              </div>
              <div className="text-center">
                <div className="w-12 h-12 bg-blue-600 text-white rounded-full flex items-center justify-center mx-auto mb-3">
                  <Brain className="w-6 h-6" />
                </div>
                <h4 className="font-medium text-blue-800 mb-2">2. Train the Model</h4>
                <p className="text-sm text-blue-600">
                  Our AI analyzes your data patterns to learn your specific waste trends and environmental factors
                </p>
              </div>
              <div className="text-center">
                <div className="w-12 h-12 bg-blue-600 text-white rounded-full flex items-center justify-center mx-auto mb-3">
                  <Percent className="w-6 h-6" />
                </div>
                <h4 className="font-medium text-blue-800 mb-2">3. Get Discount Recommendations</h4>
                <p className="text-sm text-blue-600">
                  Receive AI-powered discount recommendations based on your trained model to prevent waste
                </p>
              </div>
            </div>
          </div>
        </>
      ) : (
        /* Discount Recommendations Tab */
        <DiscountRecommendations trainedModelData={dataSummary} />
      )}
    </div>
  );
};

export default DataTraining;