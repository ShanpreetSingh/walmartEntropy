// Training data for SmartWaste AI prediction models
// This simulates real-world data that would be collected from stores

export interface ProductWasteData {
  productId: string;
  productName: string;
  category: string;
  purchaseDate: string;
  expiryDate: string;
  currentStock: number;
  dailySales: number[];
  temperature: number[];
  humidity: number[];
  location: string;
  seasonality: number;
  promotionActive: boolean;
  wasteAmount: number;
  actualSoldAmount: number;
}

export interface EnvironmentalFactors {
  date: string;
  temperature: number;
  humidity: number;
  footTraffic: number;
  weatherCondition: string;
  isHoliday: boolean;
  isWeekend: boolean;
}

export interface SalesPattern {
  productCategory: string;
  hourlyPattern: number[]; // 24 hours
  weeklyPattern: number[]; // 7 days
  monthlyPattern: number[]; // 12 months
  seasonalMultiplier: number;
}

// Historical waste data for training ML models
export const historicalWasteData: ProductWasteData[] = [
  {
    productId: "1",
    productName: "Organic Bananas",
    category: "produce",
    purchaseDate: "2024-01-15",
    expiryDate: "2024-01-22",
    currentStock: 45,
    dailySales: [12, 8, 15, 6, 9, 11, 7],
    temperature: [68, 69, 67, 70, 68, 69, 71],
    humidity: [65, 67, 63, 68, 66, 64, 69],
    location: "Produce Section A",
    seasonality: 0.8,
    promotionActive: false,
    wasteAmount: 8,
    actualSoldAmount: 37
  },
  {
    productId: "2",
    productName: "Fresh Milk",
    category: "dairy",
    purchaseDate: "2024-01-10",
    expiryDate: "2024-01-17",
    currentStock: 24,
    dailySales: [18, 22, 16, 20, 19, 25, 15],
    temperature: [38, 39, 37, 38, 39, 38, 40],
    humidity: [45, 47, 44, 46, 45, 48, 43],
    location: "Dairy Section B",
    seasonality: 1.0,
    promotionActive: true,
    wasteAmount: 2,
    actualSoldAmount: 22
  },
  {
    productId: "3",
    productName: "Premium Bread",
    category: "bakery",
    purchaseDate: "2024-01-12",
    expiryDate: "2024-01-15",
    currentStock: 18,
    dailySales: [8, 6, 12, 4, 7, 9, 5],
    temperature: [72, 73, 71, 74, 72, 73, 75],
    humidity: [55, 57, 54, 58, 56, 55, 59],
    location: "Bakery Section",
    seasonality: 0.9,
    promotionActive: false,
    wasteAmount: 5,
    actualSoldAmount: 13
  },
  {
    productId: "4",
    productName: "Greek Yogurt",
    category: "dairy",
    purchaseDate: "2024-01-08",
    expiryDate: "2024-01-20",
    currentStock: 32,
    dailySales: [14, 16, 12, 18, 15, 20, 11],
    temperature: [38, 37, 39, 38, 37, 38, 39],
    humidity: [46, 48, 45, 47, 46, 49, 44],
    location: "Dairy Section A",
    seasonality: 1.1,
    promotionActive: true,
    wasteAmount: 3,
    actualSoldAmount: 29
  },
  {
    productId: "5",
    productName: "Mixed Vegetables",
    category: "produce",
    purchaseDate: "2024-01-14",
    expiryDate: "2024-01-18",
    currentStock: 28,
    dailySales: [10, 8, 14, 6, 9, 12, 7],
    temperature: [68, 69, 67, 70, 68, 69, 71],
    humidity: [64, 66, 62, 67, 65, 63, 68],
    location: "Produce Section B",
    seasonality: 0.7,
    promotionActive: false,
    wasteAmount: 6,
    actualSoldAmount: 22
  }
];

// Environmental factors affecting product shelf life
export const environmentalData: EnvironmentalFactors[] = [
  {
    date: "2024-01-15",
    temperature: 72,
    humidity: 65,
    footTraffic: 1250,
    weatherCondition: "sunny",
    isHoliday: false,
    isWeekend: false
  },
  {
    date: "2024-01-16",
    temperature: 74,
    humidity: 68,
    footTraffic: 980,
    weatherCondition: "cloudy",
    isHoliday: false,
    isWeekend: false
  },
  {
    date: "2024-01-17",
    temperature: 71,
    humidity: 63,
    footTraffic: 1450,
    weatherCondition: "rainy",
    isHoliday: false,
    isWeekend: false
  },
  {
    date: "2024-01-18",
    temperature: 73,
    humidity: 66,
    footTraffic: 1680,
    weatherCondition: "sunny",
    isHoliday: false,
    isWeekend: false
  },
  {
    date: "2024-01-19",
    temperature: 75,
    humidity: 70,
    footTraffic: 1890,
    weatherCondition: "sunny",
    isHoliday: false,
    isWeekend: true
  },
  {
    date: "2024-01-20",
    temperature: 76,
    humidity: 72,
    footTraffic: 2100,
    weatherCondition: "sunny",
    isHoliday: false,
    isWeekend: true
  }
];

// Sales patterns for different product categories
export const salesPatterns: SalesPattern[] = [
  {
    productCategory: "produce",
    hourlyPattern: [2, 1, 1, 1, 2, 4, 8, 12, 15, 18, 20, 22, 25, 20, 18, 16, 14, 12, 10, 8, 6, 4, 3, 2],
    weeklyPattern: [85, 90, 88, 92, 95, 100, 78], // Mon-Sun, normalized to 100
    monthlyPattern: [90, 85, 95, 100, 105, 110, 108, 106, 102, 98, 88, 92],
    seasonalMultiplier: 1.0
  },
  {
    productCategory: "dairy",
    hourlyPattern: [3, 2, 1, 1, 2, 5, 10, 15, 18, 20, 22, 24, 26, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 3],
    weeklyPattern: [88, 92, 90, 94, 96, 100, 82],
    monthlyPattern: [95, 90, 98, 102, 100, 105, 103, 101, 99, 96, 85, 88],
    seasonalMultiplier: 1.0
  },
  {
    productCategory: "bakery",
    hourlyPattern: [1, 1, 1, 1, 3, 8, 15, 20, 25, 22, 18, 16, 20, 18, 15, 12, 10, 8, 6, 4, 3, 2, 1, 1],
    weeklyPattern: [80, 85, 83, 88, 92, 100, 75],
    monthlyPattern: [88, 82, 90, 95, 98, 102, 100, 98, 95, 92, 78, 85],
    seasonalMultiplier: 0.9
  },
  {
    productCategory: "meat",
    hourlyPattern: [2, 1, 1, 1, 2, 6, 12, 18, 22, 25, 28, 30, 32, 28, 25, 22, 20, 18, 15, 12, 8, 5, 3, 2],
    weeklyPattern: [90, 95, 92, 96, 98, 100, 85],
    monthlyPattern: [92, 88, 95, 100, 102, 108, 106, 104, 100, 98, 82, 90],
    seasonalMultiplier: 1.1
  }
];

// ML Model Training Configuration
export const modelConfig = {
  features: [
    'currentStock',
    'dailySalesAvg',
    'daysUntilExpiry',
    'temperature',
    'humidity',
    'seasonality',
    'promotionActive',
    'footTraffic',
    'isWeekend',
    'isHoliday',
    'hourOfDay',
    'dayOfWeek'
  ],
  target: 'wasteAmount',
  algorithms: [
    'randomForest',
    'xgboost',
    'linearRegression',
    'neuralNetwork'
  ],
  validationSplit: 0.2,
  crossValidationFolds: 5
};

// Prediction thresholds for different risk levels
export const riskThresholds = {
  low: { wastePercentage: 0, color: '#00a651' },
  medium: { wastePercentage: 15, color: '#ffc220' },
  high: { wastePercentage: 30, color: '#ff6b35' },
  critical: { wastePercentage: 50, color: '#e60012' }
};

// NGO Integration Data
export const ngoPartners = [
  {
    id: 'ngo1',
    name: 'Food Bank Plus',
    acceptedCategories: ['produce', 'dairy', 'bakery', 'canned'],
    maxCapacityPerDay: 500, // pounds
    pickupSchedule: ['monday', 'wednesday', 'friday'],
    contactInfo: {
      phone: '+1-555-123-4567',
      email: 'donations@foodbankplus.org',
      address: '123 Community St, City, State 12345'
    },
    certifications: ['FDA', 'USDA', 'Local Health Dept'],
    averageResponseTime: 2 // hours
  },
  {
    id: 'ngo2',
    name: 'Community Kitchen Network',
    acceptedCategories: ['produce', 'meat', 'dairy'],
    maxCapacityPerDay: 300,
    pickupSchedule: ['tuesday', 'thursday', 'saturday'],
    contactInfo: {
      phone: '+1-555-987-6543',
      email: 'help@communitykitchen.org',
      address: '456 Helper Ave, City, State 12345'
    },
    certifications: ['FDA', 'Local Health Dept'],
    averageResponseTime: 3
  }
];