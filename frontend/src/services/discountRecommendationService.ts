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

interface DiscountDecision {
  recommendationId: string;
  decision: 'approved' | 'rejected';
  actualDiscountApplied?: number;
  ownerNotes?: string;
  timestamp: Date;
}

export class DiscountRecommendationService {
  private static instance: DiscountRecommendationService;
  private recommendations: DiscountRecommendation[] = [];
  private decisions: DiscountDecision[] = [];
  private trainedModelData: any = null;
  private userTrainingData: any[] = [];

  static getInstance(): DiscountRecommendationService {
    if (!DiscountRecommendationService.instance) {
      DiscountRecommendationService.instance = new DiscountRecommendationService();
    }
    return DiscountRecommendationService.instance;
  }

  setTrainedModelData(modelData: any, rawTrainingData: any[]): void {
    this.trainedModelData = modelData;
    this.userTrainingData = rawTrainingData;
    console.log('Set trained model data:', modelData);
    console.log('Raw training data points:', rawTrainingData.length);
  }

  private analyzeHistoricalWastePattern(productName: string, category: string): {
    wasteRate: number;
    dataPoints: number;
    avgDaysToWaste: number;
    environmentalFactors: any;
  } {
    // Find exact product matches first
    const exactMatches = this.userTrainingData.filter(d => 
      d.productName.toLowerCase() === productName.toLowerCase()
    );

    // If no exact matches, find category matches
    const categoryMatches = this.userTrainingData.filter(d => 
      d.category.toLowerCase() === category.toLowerCase()
    );

    const relevantData = exactMatches.length > 0 ? exactMatches : categoryMatches;

    if (relevantData.length === 0) {
      return { wasteRate: 0, dataPoints: 0, avgDaysToWaste: 0, environmentalFactors: {} };
    }

    // Calculate waste rate from historical data
    const totalInitialStock = relevantData.reduce((sum, d) => sum + d.initialStock, 0);
    const totalWaste = relevantData.reduce((sum, d) => sum + d.wasteAmount, 0);
    const wasteRate = totalInitialStock > 0 ? (totalWaste / totalInitialStock) * 100 : 0;

    // Calculate average days from purchase to waste
    const avgDaysToWaste = relevantData.reduce((sum, d) => {
      const purchaseDate = new Date(d.purchaseDate);
      const expiryDate = new Date(d.expiryDate);
      const daysDiff = (expiryDate.getTime() - purchaseDate.getTime()) / (1000 * 60 * 60 * 24);
      return sum + daysDiff;
    }, 0) / relevantData.length;

    // Analyze environmental factors
    const avgTemp = relevantData.reduce((sum, d) => sum + d.temperature, 0) / relevantData.length;
    const avgHumidity = relevantData.reduce((sum, d) => sum + d.humidity, 0) / relevantData.length;

    return {
      wasteRate,
      dataPoints: relevantData.length,
      avgDaysToWaste,
      environmentalFactors: { avgTemp, avgHumidity }
    };
  }

  private calculatePredictedWaste(
    currentStock: number,
    hoursUntilExpiry: number,
    historicalPattern: any,
    currentConditions: any
  ): { wasteAmount: number; wastePercentage: number; confidence: number } {
    
    let baseWasteRate = historicalPattern.wasteRate;
    
    // Adjust based on time until expiry
    const daysUntilExpiry = hoursUntilExpiry / 24;
    if (daysUntilExpiry <= 1) {
      baseWasteRate *= 2.0; // Double waste rate if expiring within 24 hours
    } else if (daysUntilExpiry <= 2) {
      baseWasteRate *= 1.5; // 50% increase if expiring within 48 hours
    } else if (daysUntilExpiry <= 3) {
      baseWasteRate *= 1.2; // 20% increase if expiring within 72 hours
    }

    // Adjust based on environmental conditions vs historical averages
    if (currentConditions.temperature > historicalPattern.environmentalFactors.avgTemp + 5) {
      baseWasteRate *= 1.3; // Higher temperature increases waste
    }
    if (currentConditions.humidity > historicalPattern.environmentalFactors.avgHumidity + 10) {
      baseWasteRate *= 1.2; // Higher humidity increases waste
    }

    // Adjust based on stock level (higher stock = higher waste risk)
    if (currentStock > 50) {
      baseWasteRate *= 1.1;
    } else if (currentStock > 100) {
      baseWasteRate *= 1.2;
    }

    const finalWastePercentage = Math.min(100, Math.max(0, baseWasteRate));
    const wasteAmount = Math.round((finalWastePercentage / 100) * currentStock);

    // Calculate confidence based on data quality
    let confidence = 50; // Base confidence
    confidence += Math.min(30, historicalPattern.dataPoints * 5); // More data = higher confidence
    confidence += hoursUntilExpiry <= 24 ? 15 : 0; // Higher confidence for urgent items
    confidence = Math.min(95, confidence);

    return {
      wasteAmount,
      wastePercentage: Math.round(finalWastePercentage * 10) / 10,
      confidence: Math.round(confidence)
    };
  }

  private calculateOptimalDiscount(
    wastePercentage: number,
    hoursUntilExpiry: number,
    historicalPattern: any
  ): number {
    let baseDiscount = 0;
    
    // Base discount based on predicted waste percentage
    if (wastePercentage >= 60) baseDiscount = 50;
    else if (wastePercentage >= 40) baseDiscount = 35;
    else if (wastePercentage >= 25) baseDiscount = 25;
    else if (wastePercentage >= 15) baseDiscount = 15;
    else baseDiscount = 10;

    // Urgency multiplier
    if (hoursUntilExpiry <= 6) baseDiscount *= 1.4;
    else if (hoursUntilExpiry <= 12) baseDiscount *= 1.2;
    else if (hoursUntilExpiry <= 24) baseDiscount *= 1.1;

    // Historical success factor (if we had previous successful discounts for this category)
    if (historicalPattern.dataPoints >= 5) {
      baseDiscount *= 1.1; // Slight increase if we have good historical data
    }

    return Math.min(70, Math.max(5, Math.round(baseDiscount)));
  }

  private generateReasoning(
    recommendation: Partial<DiscountRecommendation>,
    historicalPattern: any
  ): string[] {
    const reasons: string[] = [];
    
    // Historical data reasoning
    if (historicalPattern.dataPoints > 0) {
      reasons.push(`Based on ${historicalPattern.dataPoints} historical data points for ${recommendation.category}`);
      reasons.push(`Historical waste rate: ${historicalPattern.wasteRate.toFixed(1)}% for this category`);
    }

    // Waste prediction reasoning
    if (recommendation.predictedWastePercentage! >= 40) {
      reasons.push(`High waste risk: ${recommendation.predictedWastePercentage}% predicted waste`);
    } else if (recommendation.predictedWastePercentage! >= 20) {
      reasons.push(`Moderate waste risk: ${recommendation.predictedWastePercentage}% predicted waste`);
    }

    // Time urgency
    if (recommendation.hoursUntilExpiry! <= 12) {
      reasons.push(`Urgent: Only ${recommendation.hoursUntilExpiry} hours until expiry`);
    } else if (recommendation.hoursUntilExpiry! <= 24) {
      reasons.push(`Time-sensitive: ${recommendation.hoursUntilExpiry} hours until expiry`);
    }

    // Stock level
    if (recommendation.currentStock! > 30) {
      reasons.push(`High inventory: ${recommendation.currentStock} units need to be moved quickly`);
    }

    return reasons;
  }

  private getRiskLevel(wastePercentage: number): 'low' | 'medium' | 'high' | 'critical' {
    if (wastePercentage >= 50) return 'critical';
    if (wastePercentage >= 30) return 'high';
    if (wastePercentage >= 15) return 'medium';
    return 'low';
  }

  generateRecommendationsFromTrainingData(): DiscountRecommendation[] {
    if (!this.trainedModelData || !this.userTrainingData || this.userTrainingData.length === 0) {
      throw new Error('No trained model data available. Please train the model first.');
    }

    const newRecommendations: DiscountRecommendation[] = [];

    // Create current inventory scenarios based on your training data
    // This simulates products that are currently in stock and approaching expiry
    const uniqueProducts = [...new Set(this.userTrainingData.map(d => d.productName))];
    
    uniqueProducts.forEach(productName => {
      const productData = this.userTrainingData.find(d => d.productName === productName);
      if (!productData) return;

      // Simulate current stock scenario (this would come from your real inventory system)
      const currentStock = Math.floor(Math.random() * 80) + 20; // 20-100 units
      const hoursUntilExpiry = Math.floor(Math.random() * 72) + 6; // 6-78 hours
      const currentPrice = Math.round((Math.random() * 15 + 2) * 100) / 100; // $2-17
      
      // Analyze historical pattern for this specific product/category
      const historicalPattern = this.analyzeHistoricalWastePattern(productName, productData.category);
      
      // Only generate recommendation if there's historical data and some waste risk
      if (historicalPattern.dataPoints > 0 && historicalPattern.wasteRate > 5) {
        
        const currentConditions = {
          temperature: productData.temperature + (Math.random() * 10 - 5), // ±5 degrees variation
          humidity: productData.humidity + (Math.random() * 20 - 10) // ±10% variation
        };

        const prediction = this.calculatePredictedWaste(
          currentStock,
          hoursUntilExpiry,
          historicalPattern,
          currentConditions
        );

        // Only recommend if waste risk is significant (>10%) or expiring soon (<24h)
        if (prediction.wastePercentage > 10 || hoursUntilExpiry <= 24) {
          const suggestedDiscount = this.calculateOptimalDiscount(
            prediction.wastePercentage,
            hoursUntilExpiry,
            historicalPattern
          );

          const discountedPrice = currentPrice * (1 - suggestedDiscount / 100);
          const estimatedRevenueSaved = prediction.wasteAmount * discountedPrice;

          const recommendation: DiscountRecommendation = {
            id: `rec_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            productName,
            category: productData.category,
            currentStock,
            hoursUntilExpiry,
            predictedWasteAmount: prediction.wasteAmount,
            predictedWastePercentage: prediction.wastePercentage,
            suggestedDiscountPercentage: suggestedDiscount,
            estimatedRevenueSaved: Math.round(estimatedRevenueSaved * 100) / 100,
            confidence: prediction.confidence,
            riskLevel: this.getRiskLevel(prediction.wastePercentage),
            reasoning: [],
            location: productData.location || `${productData.category} Section`,
            currentPrice,
            discountedPrice: Math.round(discountedPrice * 100) / 100,
            status: 'pending',
            timestamp: new Date(),
            basedOnDataPoints: historicalPattern.dataPoints,
            historicalWasteRate: Math.round(historicalPattern.wasteRate * 10) / 10
          };

          recommendation.reasoning = this.generateReasoning(recommendation, historicalPattern);
          newRecommendations.push(recommendation);
        }
      }
    });

    // Sort by urgency (combination of waste risk and time until expiry)
    newRecommendations.sort((a, b) => {
      const urgencyA = (a.predictedWastePercentage / 100) * (72 - a.hoursUntilExpiry) / 72;
      const urgencyB = (b.predictedWastePercentage / 100) * (72 - b.hoursUntilExpiry) / 72;
      return urgencyB - urgencyA;
    });

    // Clear old pending recommendations and add new ones
    this.recommendations = this.recommendations.filter(r => r.status !== 'pending');
    this.recommendations = [...this.recommendations, ...newRecommendations];

    console.log(`Generated ${newRecommendations.length} recommendations based on your training data`);
    return newRecommendations;
  }

  approveRecommendation(recommendationId: string, actualDiscount?: number, notes?: string): void {
    const recommendation = this.recommendations.find(r => r.id === recommendationId);
    if (recommendation) {
      recommendation.status = 'approved';
      if (actualDiscount) {
        recommendation.suggestedDiscountPercentage = actualDiscount;
        recommendation.discountedPrice = recommendation.currentPrice * (1 - actualDiscount / 100);
        recommendation.estimatedRevenueSaved = recommendation.predictedWasteAmount * recommendation.discountedPrice;
      }
    }

    const decision: DiscountDecision = {
      recommendationId,
      decision: 'approved',
      actualDiscountApplied: actualDiscount || recommendation?.suggestedDiscountPercentage,
      ownerNotes: notes,
      timestamp: new Date()
    };

    this.decisions.push(decision);
    console.log(`Discount approved for ${recommendation?.productName}: ${actualDiscount || recommendation?.suggestedDiscountPercentage}%`);
  }

  rejectRecommendation(recommendationId: string, notes?: string): void {
    const recommendation = this.recommendations.find(r => r.id === recommendationId);
    if (recommendation) {
      recommendation.status = 'rejected';
    }

    const decision: DiscountDecision = {
      recommendationId,
      decision: 'rejected',
      ownerNotes: notes,
      timestamp: new Date()
    };

    this.decisions.push(decision);
    console.log(`Discount rejected for ${recommendation?.productName}`);
  }

  getPendingRecommendations(): DiscountRecommendation[] {
    return this.recommendations.filter(r => r.status === 'pending');
  }

  getAllRecommendations(): DiscountRecommendation[] {
    return this.recommendations.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
  }

  getDecisionHistory(): DiscountDecision[] {
    return this.decisions.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
  }

  getRecommendationStats() {
    const total = this.recommendations.length;
    const approved = this.recommendations.filter(r => r.status === 'approved').length;
    const rejected = this.recommendations.filter(r => r.status === 'rejected').length;
    const pending = this.recommendations.filter(r => r.status === 'pending').length;
    
    const totalPotentialSavings = this.recommendations
      .filter(r => r.status === 'approved')
      .reduce((sum, r) => sum + r.estimatedRevenueSaved, 0);

    return {
      total,
      approved,
      rejected,
      pending,
      approvalRate: total > 0 ? Math.round((approved / total) * 100) : 0,
      totalPotentialSavings: Math.round(totalPotentialSavings * 100) / 100
    };
  }

  clearRecommendations(): void {
    this.recommendations = [];
    this.decisions = [];
  }
}