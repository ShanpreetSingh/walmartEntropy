interface UserTrainingData {
  productName: string;
  category: string;
  purchaseDate: string;
  expiryDate: string;
  initialStock: number;
  finalStock: number;
  wasteAmount: number;
  temperature: number;
  humidity: number;
  promotionActive: boolean;
  location: string;
}

interface TrainingSession {
  id: string;
  timestamp: Date;
  dataPoints: UserTrainingData[];
  modelAccuracy: number;
  status: 'training' | 'completed' | 'failed';
}

export class UserTrainingService {
  private static instance: UserTrainingService;
  private trainingData: UserTrainingData[] = [];
  private trainingSessions: TrainingSession[] = [];

  static getInstance(): UserTrainingService {
    if (!UserTrainingService.instance) {
      UserTrainingService.instance = new UserTrainingService();
    }
    return UserTrainingService.instance;
  }

  addTrainingData(data: UserTrainingData[]): void {
    this.trainingData = [...this.trainingData, ...data];
    console.log(`Added ${data.length} training data points. Total: ${this.trainingData.length}`);
  }

  clearTrainingData(): void {
    this.trainingData = [];
    this.trainingSessions = [];
    console.log('Training data cleared');
  }

  getTrainingData(): UserTrainingData[] {
    return this.trainingData;
  }

  getDataSummary() {
    const categories = [...new Set(this.trainingData.map(d => d.category))];
    const totalWaste = this.trainingData.reduce((sum, d) => sum + d.wasteAmount, 0);
    const totalStock = this.trainingData.reduce((sum, d) => sum + d.initialStock, 0);
    const wastePercentage = totalStock > 0 ? (totalWaste / totalStock) * 100 : 0;

    return {
      totalRecords: this.trainingData.length,
      categories: categories.length,
      totalWaste,
      totalStock,
      wastePercentage: Math.round(wastePercentage * 10) / 10,
      categoryBreakdown: categories.map(cat => ({
        category: cat,
        count: this.trainingData.filter(d => d.category === cat).length,
        waste: this.trainingData.filter(d => d.category === cat).reduce((sum, d) => sum + d.wasteAmount, 0)
      }))
    };
  }

  async trainModel(): Promise<TrainingSession> {
    if (this.trainingData.length === 0) {
      throw new Error('No training data available. Please upload data first.');
    }

    const sessionId = `session_${Date.now()}`;
    const session: TrainingSession = {
      id: sessionId,
      timestamp: new Date(),
      dataPoints: [...this.trainingData],
      modelAccuracy: 0,
      status: 'training'
    };

    this.trainingSessions.push(session);

    // Simulate training process
    return new Promise((resolve) => {
      setTimeout(() => {
        // Calculate mock accuracy based on data quality
        const dataQuality = Math.min(100, (this.trainingData.length / 100) * 100);
        const baseAccuracy = 65;
        const qualityBonus = (dataQuality / 100) * 25;
        const randomVariation = Math.random() * 10;
        
        session.modelAccuracy = Math.min(95, baseAccuracy + qualityBonus + randomVariation);
        session.status = 'completed';
        
        console.log(`Training completed with ${session.modelAccuracy.toFixed(1)}% accuracy`);
        console.log(`Model trained on ${this.trainingData.length} data points`);
        resolve(session);
      }, 3000);
    });
  }

  getTrainingSessions(): TrainingSession[] {
    return this.trainingSessions.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
  }

  getLatestSession(): TrainingSession | null {
    const sessions = this.getTrainingSessions();
    return sessions.length > 0 ? sessions[0] : null;
  }

  validateData(data: UserTrainingData[]): { valid: boolean; errors: string[] } {
    const errors: string[] = [];

    data.forEach((row, index) => {
      if (!row.productName.trim()) {
        errors.push(`Row ${index + 1}: Product name is required`);
      }
      
      if (!row.purchaseDate || !row.expiryDate) {
        errors.push(`Row ${index + 1}: Purchase and expiry dates are required`);
      }
      
      if (new Date(row.purchaseDate) >= new Date(row.expiryDate)) {
        errors.push(`Row ${index + 1}: Expiry date must be after purchase date`);
      }
      
      if (row.initialStock < 0 || row.finalStock < 0 || row.wasteAmount < 0) {
        errors.push(`Row ${index + 1}: Stock and waste amounts cannot be negative`);
      }
      
      if (row.wasteAmount > row.initialStock) {
        errors.push(`Row ${index + 1}: Waste amount cannot exceed initial stock`);
      }
      
      if (row.temperature < -20 || row.temperature > 120) {
        errors.push(`Row ${index + 1}: Temperature seems unrealistic`);
      }
      
      if (row.humidity < 0 || row.humidity > 100) {
        errors.push(`Row ${index + 1}: Humidity must be between 0-100%`);
      }
    });

    return {
      valid: errors.length === 0,
      errors
    };
  }

  exportTrainingData(): string {
    const headers = [
      'Product Name', 'Category', 'Purchase Date', 'Expiry Date',
      'Initial Stock', 'Final Stock', 'Waste Amount', 'Temperature',
      'Humidity', 'Promotion Active', 'Location'
    ];

    const csvContent = [
      headers.join(','),
      ...this.trainingData.map(row => [
        row.productName,
        row.category,
        row.purchaseDate,
        row.expiryDate,
        row.initialStock,
        row.finalStock,
        row.wasteAmount,
        row.temperature,
        row.humidity,
        row.promotionActive,
        row.location
      ].join(','))
    ].join('\n');

    return csvContent;
  }

  // Get specific product data for analysis
  getProductData(productName: string): UserTrainingData[] {
    return this.trainingData.filter(d => 
      d.productName.toLowerCase() === productName.toLowerCase()
    );
  }

  // Get category data for analysis
  getCategoryData(category: string): UserTrainingData[] {
    return this.trainingData.filter(d => 
      d.category.toLowerCase() === category.toLowerCase()
    );
  }

  // Calculate waste rate for a specific product or category
  calculateWasteRate(productName?: string, category?: string): number {
    let relevantData = this.trainingData;
    
    if (productName) {
      relevantData = this.getProductData(productName);
    } else if (category) {
      relevantData = this.getCategoryData(category);
    }

    if (relevantData.length === 0) return 0;

    const totalInitialStock = relevantData.reduce((sum, d) => sum + d.initialStock, 0);
    const totalWaste = relevantData.reduce((sum, d) => sum + d.wasteAmount, 0);
    
    return totalInitialStock > 0 ? (totalWaste / totalInitialStock) * 100 : 0;
  }
}