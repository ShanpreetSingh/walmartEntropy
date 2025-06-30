import { mockProducts } from '../data/mockData';

export interface ChatResponse {
  message: string;
  products?: Array<{
    id: string;
    name: string;
    price: number;
    image: string;
    category: string;
    deal?: {
      discount: number;
      originalPrice: number;
    };
  }>;
  suggestions?: string[];
}

interface UserPreferences {
  budget: string;
  categories: string[];
  dietaryRestrictions: string[];
}

interface ConversationMemory {
  userInterests: string[];
  preferredCategories: string[];
  priceRange: { min: number; max: number } | null;
  previousPurchases: string[];
  sessionStartTime: Date;
}

export class ChatbotService {
  private static instance: ChatbotService;
  private conversationHistory: string[] = [];
  private conversationMemory: ConversationMemory;
  private feedbackData: { [messageId: string]: 'positive' | 'negative' } = {};
  private userEngagement = {
    totalMessages: 0,
    productsViewed: 0,
    productsAddedToCart: 0,
    averageResponseTime: 0
  };

  constructor() {
    this.conversationMemory = {
      userInterests: [],
      preferredCategories: [],
      priceRange: null,
      previousPurchases: [],
      sessionStartTime: new Date()
    };
  }

  static getInstance(): ChatbotService {
    if (!ChatbotService.instance) {
      ChatbotService.instance = new ChatbotService();
    }
    return ChatbotService.instance;
  }

  private analyzeIntent(message: string, context: string[] = []): {
    intent: string;
    entities: string[];
    confidence: number;
    sentiment: 'positive' | 'neutral' | 'negative';
    urgency: 'low' | 'medium' | 'high';
  } {
    const lowerMessage = message.toLowerCase();
    
    // Enhanced category detection with synonyms
    const categories = {
      groceries: [
        'grocery', 'groceries', 'food', 'eat', 'cooking', 'kitchen', 'fresh', 'organic', 
        'fruit', 'vegetable', 'meat', 'dairy', 'bread', 'milk', 'cheese', 'snack', 
        'beverage', 'drink', 'produce', 'frozen', 'canned', 'pantry', 'breakfast',
        'lunch', 'dinner', 'healthy', 'nutrition'
      ],
      electronics: [
        'electronics', 'electronic', 'phone', 'smartphone', 'laptop', 'computer', 
        'headphones', 'tech', 'gadget', 'device', 'tablet', 'tv', 'television',
        'camera', 'gaming', 'console', 'smart', 'wireless', 'bluetooth', 'charger'
      ],
      clothing: [
        'clothing', 'clothes', 'shirt', 'pants', 'dress', 'shoes', 'fashion', 
        'wear', 'outfit', 'jacket', 'jeans', 'sweater', 'boots', 'sneakers',
        'accessories', 'hat', 'belt', 'style', 'size', 'color'
      ],
      home: [
        'home', 'house', 'furniture', 'decor', 'kitchen', 'bedroom', 'living room', 
        'appliance', 'lamp', 'blender', 'cleaning', 'organization', 'storage',
        'bathroom', 'garden', 'outdoor', 'tools', 'improvement'
      ]
    };

    // Enhanced intent detection with context awareness
    const intents = {
      search_product: [
        'looking for', 'need', 'want', 'find', 'search', 'show me', 'get me',
        'where can i find', 'do you have', 'i\'m looking', 'help me find'
      ],
      price_inquiry: [
        'cheap', 'expensive', 'budget', 'affordable', 'cost', 'price', 'money',
        'under', 'less than', 'maximum', 'minimum', 'range', 'how much'
      ],
      deal_inquiry: [
        'deal', 'deals', 'discount', 'sale', 'offer', 'promotion', 'bargain',
        'special', 'clearance', 'markdown', 'reduced', 'savings', 'coupon'
      ],
      recommendation: [
        'recommend', 'suggest', 'advice', 'help', 'what should', 'best',
        'popular', 'trending', 'top rated', 'favorite', 'good choice'
      ],
      comparison: [
        'compare', 'difference', 'better', 'versus', 'vs', 'which one',
        'pros and cons', 'similar', 'alternative', 'option'
      ],
      greeting: [
        'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
        'howdy', 'greetings', 'what\'s up'
      ],
      waste_reduction: [
        'eco', 'environment', 'waste', 'sustainable', 'green', 'expiring',
        'organic', 'natural', 'recyclable', 'biodegradable', 'earth friendly'
      ],
      complaint: [
        'problem', 'issue', 'wrong', 'broken', 'defective', 'complaint',
        'disappointed', 'unsatisfied', 'return', 'refund'
      ],
      compliment: [
        'great', 'awesome', 'excellent', 'perfect', 'amazing', 'wonderful',
        'fantastic', 'love', 'thank you', 'thanks', 'helpful'
      ]
    };

    let detectedIntent = 'general';
    let confidence = 0.5;
    let sentiment: 'positive' | 'neutral' | 'negative' = 'neutral';
    let urgency: 'low' | 'medium' | 'high' = 'low';
    const entities: string[] = [];

    // Detect intent with context awareness
    for (const [intent, keywords] of Object.entries(intents)) {
      const matches = keywords.filter(keyword => lowerMessage.includes(keyword));
      if (matches.length > 0) {
        detectedIntent = intent;
        confidence = Math.min(0.95, 0.6 + (matches.length * 0.1));
        
        // Boost confidence if context supports the intent
        if (context.length > 0) {
          const contextSupport = context.some(ctx => 
            keywords.some(keyword => ctx.toLowerCase().includes(keyword))
          );
          if (contextSupport) confidence += 0.1;
        }
        break;
      }
    }

    // Detect categories
    for (const [category, keywords] of Object.entries(categories)) {
      const matches = keywords.filter(keyword => lowerMessage.includes(keyword));
      if (matches.length > 0) {
        entities.push(category);
        this.updateUserInterests(category);
      }
    }

    // Sentiment analysis
    const positiveWords = ['good', 'great', 'excellent', 'love', 'like', 'awesome', 'perfect', 'amazing'];
    const negativeWords = ['bad', 'terrible', 'hate', 'dislike', 'awful', 'horrible', 'worst', 'disappointed'];
    
    const positiveCount = positiveWords.filter(word => lowerMessage.includes(word)).length;
    const negativeCount = negativeWords.filter(word => lowerMessage.includes(word)).length;
    
    if (positiveCount > negativeCount) sentiment = 'positive';
    else if (negativeCount > positiveCount) sentiment = 'negative';

    // Urgency detection
    const urgentWords = ['urgent', 'asap', 'immediately', 'now', 'quick', 'fast', 'emergency'];
    const mediumUrgencyWords = ['soon', 'today', 'this week', 'need'];
    
    if (urgentWords.some(word => lowerMessage.includes(word))) urgency = 'high';
    else if (mediumUrgencyWords.some(word => lowerMessage.includes(word))) urgency = 'medium';

    // Extract price range
    const priceMatch = lowerMessage.match(/under \$?(\d+)|less than \$?(\d+)|maximum \$?(\d+)|max \$?(\d+)/);
    if (priceMatch) {
      const maxPrice = parseInt(priceMatch[1] || priceMatch[2] || priceMatch[3] || priceMatch[4]);
      this.conversationMemory.priceRange = { min: 0, max: maxPrice };
    }

    return { intent: detectedIntent, entities, confidence, sentiment, urgency };
  }

  private updateUserInterests(category: string): void {
    if (!this.conversationMemory.userInterests.includes(category)) {
      this.conversationMemory.userInterests.push(category);
    }
    if (!this.conversationMemory.preferredCategories.includes(category)) {
      this.conversationMemory.preferredCategories.push(category);
    }
  }

  private getPersonalizedProducts(category: string, limit: number = 4, priceRange?: { min: number; max: number }) {
    let filteredProducts = mockProducts.filter(product => product.category === category);
    
    // Apply price filter if specified
    if (priceRange) {
      filteredProducts = filteredProducts.filter(product => 
        product.price >= priceRange.min && product.price <= priceRange.max
      );
    }

    // Prioritize products with deals for better value
    filteredProducts.sort((a, b) => {
      const aHasDeal = a.deal ? 1 : 0;
      const bHasDeal = b.deal ? 1 : 0;
      return bHasDeal - aHasDeal;
    });

    return filteredProducts
      .slice(0, limit)
      .map(product => ({
        id: product.id,
        name: product.name,
        price: product.price,
        image: product.image,
        category: product.category,
        deal: product.deal
      }));
  }

  private getSmartRecommendations(userPreferences: UserPreferences, limit: number = 4) {
    // Get products based on conversation history and preferences
    const preferredCategories = this.conversationMemory.preferredCategories.length > 0 
      ? this.conversationMemory.preferredCategories 
      : ['groceries', 'electronics'];

    let recommendations: any[] = [];
    
    preferredCategories.forEach(category => {
          const categoryProducts = this.getPersonalizedProducts(category, 2, this.conversationMemory.priceRange ?? undefined);
          recommendations = [...recommendations, ...categoryProducts];
        });

    // If no category-specific recommendations, get trending deals
    if (recommendations.length === 0) {
      recommendations = this.getDealsProducts(limit);
    }

    return recommendations.slice(0, limit);
  }

  private getDealsProducts(limit: number = 4) {
    return mockProducts
      .filter(product => product.deal)
      .sort((a, b) => (b.deal?.discount || 0) - (a.deal?.discount || 0))
      .slice(0, limit)
      .map(product => ({
        id: product.id,
        name: product.name,
        price: product.price,
        image: product.image,
        category: product.category,
        deal: product.deal
      }));
  }

  private getBudgetProducts(limit: number = 4, maxPrice: number = 50) {
    return mockProducts
      .filter(product => product.price <= maxPrice)
      .sort((a, b) => a.price - b.price)
      .slice(0, limit)
      .map(product => ({
        id: product.id,
        name: product.name,
        price: product.price,
        image: product.image,
        category: product.category,
        deal: product.deal
      }));
  }

  private getWasteReductionProducts(limit: number = 4) {
    return mockProducts
      .filter(product => product.deal?.type === 'waste-reduction' || product.deal?.type === 'flash')
      .slice(0, limit)
      .map(product => ({
        id: product.id,
        name: product.name,
        price: product.price,
        image: product.image,
        category: product.category,
        deal: product.deal
      }));
  }

  private generateContextualSuggestions(intent: string, entities: string[], sentiment: string): string[] {
    const baseSuggestions = {
      search_product: ["Show me more options", "Compare similar items", "Filter by price", "Show customer reviews"],
      deal_inquiry: ["Flash sales", "Clearance items", "Bundle deals", "Weekly specials"],
      recommendation: ["Best sellers", "Customer favorites", "New arrivals", "Trending now"],
      price_inquiry: ["Budget under $25", "Premium options", "Best value deals", "Price comparison"],
      waste_reduction: ["Eco-friendly products", "Sustainable brands", "Organic options", "Recyclable items"],
      greeting: ["Browse categories", "Today's deals", "Popular items", "Help me find something"]
    };

    let suggestions = baseSuggestions[intent as keyof typeof baseSuggestions] || baseSuggestions.greeting;

    // Add entity-specific suggestions
    if (entities.length > 0) {
      entities.forEach(entity => {
        suggestions.push(`More ${entity} items`);
        suggestions.push(`${entity} deals`);
      });
    }

    // Add personalized suggestions based on conversation history
    if (this.conversationMemory.preferredCategories.length > 0) {
      const topCategory = this.conversationMemory.preferredCategories[0];
      suggestions.push(`Recommended ${topCategory}`);
    }

    return suggestions.slice(0, 4);
  }

  generateResponse(
    userMessage: string, 
    context: string[] = [], 
    userPreferences: UserPreferences = { budget: '', categories: [], dietaryRestrictions: [] }
  ): ChatResponse {
    this.conversationHistory.push(userMessage);
    this.userEngagement.totalMessages++;
    
    const analysis = this.analyzeIntent(userMessage, context);
    
    let response: ChatResponse = {
      message: "I understand you're looking for products. Could you be more specific about what you need?",
      suggestions: this.generateContextualSuggestions(analysis.intent, analysis.entities, analysis.sentiment)
    };

    // Handle different intents with enhanced responses
    switch (analysis.intent) {
      case 'greeting':
        const timeOfDay = new Date().getHours();
        const greeting = timeOfDay < 12 ? 'Good morning' : timeOfDay < 18 ? 'Good afternoon' : 'Good evening';
        
        response = {
          message: `${greeting}! Welcome to Walmart! I'm your AI shopping assistant powered by SmartWaste technology. I can help you find products, discover deals, and make eco-friendly choices. What brings you here today?`,
          suggestions: ["Show me today's deals", "I need groceries", "Electronics on sale", "Eco-friendly products"]
        };
        break;

      case 'search_product':
        if (analysis.entities.length > 0) {
          const category = analysis.entities[0];
          const products = this.getPersonalizedProducts(category, 4, this.conversationMemory.priceRange ?? undefined);
          
          response = {
            message: `Perfect! I found some excellent ${category} products for you${this.conversationMemory.priceRange ? ` within your budget` : ''}. Here are my top recommendations:`,
            products,
            suggestions: this.generateContextualSuggestions(analysis.intent, analysis.entities, analysis.sentiment)
          };
        } else {
          response = {
            message: "I'd love to help you find the perfect products! What category interests you most, or do you have something specific in mind?",
            suggestions: ["Groceries", "Electronics", "Clothing", "Home & Garden", "Show me deals"]
          };
        }
        break;

      case 'deal_inquiry':
        const dealProducts = this.getDealsProducts();
        response = {
          message: `Fantastic! I've found some incredible deals for you. Our SmartWaste AI has identified these special offers that help reduce waste while saving you money:`,
          products: dealProducts,
          suggestions: ["More flash sales", "Clearance items", "Eco-friendly deals", "Category-specific deals"]
        };
        break;

      case 'price_inquiry':
        const maxPrice = this.conversationMemory.priceRange?.max || 50;
        const budgetProducts = this.getBudgetProducts(4, maxPrice);
        
        response = {
          message: `Great! I've found some excellent budget-friendly options${maxPrice !== 50 ? ` under $${maxPrice}` : ''}. These products offer outstanding value:`,
          products: budgetProducts,
          suggestions: ["More budget items", "Best value deals", "Clearance section", "Bundle savings"]
        };
        break;

      case 'waste_reduction':
        const ecoProducts = this.getWasteReductionProducts();
        response = {
          message: "Excellent choice for sustainable shopping! Here are eco-friendly deals from our SmartWaste AI system. These products help reduce waste while offering great savings:",
          products: ecoProducts,
          suggestions: ["More eco deals", "Sustainable brands", "Organic products", "Learn about our impact"]
        };
        break;

      case 'recommendation':
        const smartRecommendations = this.getSmartRecommendations(userPreferences);
        response = {
          message: `Based on your interests and our conversation, here are my personalized recommendations for you:`,
          products: smartRecommendations,
          suggestions: ["More like these", "Customer favorites", "Trending items", "New arrivals"]
        };
        break;

      case 'comparison':
        response = {
          message: "I'd be happy to help you compare products! Could you tell me which specific items or categories you'd like me to compare?",
          suggestions: ["Compare phones", "Best vs budget options", "Brand comparison", "Feature comparison"]
        };
        break;

      case 'complaint':
        response = {
          message: "I'm sorry to hear you're having an issue. I want to help make this right. Could you tell me more about the problem so I can assist you better?",
          suggestions: ["Return policy", "Contact customer service", "Product replacement", "Refund information"]
        };
        break;

      case 'compliment':
        response = {
          message: "Thank you so much! I'm delighted I could help. Is there anything else you'd like to explore or any other way I can assist you today?",
          suggestions: ["Browse more products", "Check out deals", "Save favorites", "Share with friends"]
        };
        break;

      default:
        // Enhanced fallback with context awareness
        if (analysis.entities.length > 0) {
          const category = analysis.entities[0];
          const products = this.getPersonalizedProducts(category);
          response = {
            message: `I found some great ${category} products that might interest you:`,
            products,
            suggestions: this.generateContextualSuggestions('search_product', analysis.entities, analysis.sentiment)
          };
        } else if (context.length > 0) {
          // Use conversation context to provide better response
          const smartRecommendations = this.getSmartRecommendations(userPreferences);
          response = {
            message: "Based on our conversation, here are some products I think you might like:",
            products: smartRecommendations,
            suggestions: ["Tell me more", "Different category", "Price range", "Special offers"]
          };
        }
        break;
    }

    return response;
  }

  recordFeedback(messageId: string, feedback: 'positive' | 'negative'): void {
    this.feedbackData[messageId] = feedback;
    
    // Use feedback to improve future responses
    if (feedback === 'positive') {
      // Boost confidence in similar response patterns
      console.log('Positive feedback recorded - improving similar responses');
    } else {
      // Learn from negative feedback
      console.log('Negative feedback recorded - adjusting response strategy');
    }
  }

  getConversationContext(): string[] {
    return this.conversationHistory.slice(-5);
  }

  getConversationMemory(): ConversationMemory {
    return this.conversationMemory;
  }

  getUserEngagement() {
    return this.userEngagement;
  }

  clearHistory(): void {
    this.conversationHistory = [];
    this.conversationMemory = {
      userInterests: [],
      preferredCategories: [],
      priceRange: null,
      previousPurchases: [],
      sessionStartTime: new Date()
    };
    this.feedbackData = {};
  }

  // Analytics methods for improving the chatbot
  getPopularIntents(): { [intent: string]: number } {
    // This would analyze conversation history to identify popular intents
    return {
      'search_product': 45,
      'deal_inquiry': 30,
      'recommendation': 15,
      'price_inquiry': 10
    };
  }

  getConversionRate(): number {
    // Calculate how often conversations lead to actions
    return this.userEngagement.productsAddedToCart / this.userEngagement.totalMessages;
  }
}