// Mock data for the Walmart SmartWaste AI platform

export const mockProducts = [
  // Flash Deals
  {
    id: '1',
    name: 'Organic Bananas (3 lbs)',
    category: 'groceries',
    price: 2.49,
    image: 'https://images.pexels.com/photos/2872755/pexels-photo-2872755.jpeg?auto=compress&cs=tinysrgb&w=400',
    rating: 4.5,
    reviews: 234,
    deal: {
      type: 'flash' as const,
      discount: 40,
      originalPrice: 4.15,
      expiryHours: 2
    },
    wasteRisk: 'high' as const
  },
  {
    id: '2',
    name: 'Fresh Milk (1 Gallon)',
    category: 'groceries',
    price: 3.99,
    image: 'https://images.pexels.com/photos/416978/pexels-photo-416978.jpeg?auto=compress&cs=tinysrgb&w=400',
    rating: 4.7,
    reviews: 456,
    deal: {
      type: 'flash' as const,
      discount: 25,
      originalPrice: 5.32,
      expiryHours: 4
    },
    wasteRisk: 'medium' as const
  },
  {
    id: '3',
    name: 'Premium Bread Loaf',
    category: 'groceries',
    price: 2.99,
    image: 'https://images.pexels.com/photos/1640777/pexels-photo-1640777.jpeg?auto=compress&cs=tinysrgb&w=400',
    rating: 4.3,
    reviews: 189,
    deal: {
      type: 'flash' as const,
      discount: 35,
      originalPrice: 4.60,
      expiryHours: 6
    },
    wasteRisk: 'high' as const
  },
  {
    id: '4',
    name: 'Greek Yogurt (32oz)',
    category: 'groceries',
    price: 4.49,
    image: 'https://images.pexels.com/photos/1435735/pexels-photo-1435735.jpeg?auto=compress&cs=tinysrgb&w=400',
    rating: 4.6,
    reviews: 321,
    deal: {
      type: 'flash' as const,
      discount: 30,
      originalPrice: 6.41,
      expiryHours: 8
    },
    wasteRisk: 'medium' as const
  },

  // Waste Reduction Deals
  {
    id: '5',
    name: 'Mixed Vegetables (2 lbs)',
    category: 'groceries',
    price: 3.79,
    image: 'https://images.pexels.com/photos/143133/pexels-photo-143133.jpeg?auto=compress&cs=tinysrgb&w=400',
    rating: 4.4,
    reviews: 276,
    deal: {
      type: 'waste-reduction' as const,
      discount: 20,
      originalPrice: 4.74
    },
    wasteRisk: 'medium' as const
  },
  {
    id: '6',
    name: 'Chicken Breast (2 lbs)',
    category: 'groceries',
    price: 8.99,
    image: 'https://images.pexels.com/photos/616401/pexels-photo-616401.jpeg?auto=compress&cs=tinysrgb&w=400',
    rating: 4.8,
    reviews: 543,
    deal: {
      type: 'waste-reduction' as const,
      discount: 15,
      originalPrice: 10.58
    },
    wasteRisk: 'low' as const
  },
  {
    id: '7',
    name: 'Seasonal Apples (3 lbs)',
    category: 'groceries',
    price: 4.29,
    image: 'https://images.pexels.com/photos/102104/pexels-photo-102104.jpeg?auto=compress&cs=tinysrgb&w=400',
    rating: 4.5,
    reviews: 198,
    deal: {
      type: 'waste-reduction' as const,
      discount: 25,
      originalPrice: 5.72
    },
    wasteRisk: 'medium' as const
  },
  {
    id: '8',
    name: 'Cheese Slices (12oz)',
    category: 'groceries',
    price: 3.49,
    image: 'https://images.pexels.com/photos/773253/pexels-photo-773253.jpeg?auto=compress&cs=tinysrgb&w=400',
    rating: 4.2,
    reviews: 167,
    deal: {
      type: 'waste-reduction' as const,
      discount: 18,
      originalPrice: 4.26
    },
    wasteRisk: 'low' as const
  },

  // Regular Products
  {
    id: '9',
    name: 'Wireless Headphones',
    category: 'electronics',
    price: 149.99,
    image: 'https://images.pexels.com/photos/3394650/pexels-photo-3394650.jpeg?auto=compress&cs=tinysrgb&w=400',
    rating: 4.7,
    reviews: 892,
    deal: {
      type: 'regular' as const,
      discount: 10,
      originalPrice: 166.66
    }
  },
  {
    id: '10',
    name: 'Smartphone Pro Max',
    category: 'electronics',
    price: 699.99,
    image: 'https://images.pexels.com/photos/607812/pexels-photo-607812.jpeg?auto=compress&cs=tinysrgb&w=400',
    rating: 4.8,
    reviews: 1234,
    deal: {
      type: 'regular' as const,
      discount: 5,
      originalPrice: 736.83
    }
  },
  {
    id: '11',
    name: 'Cotton T-Shirt',
    category: 'clothing',
    price: 8.99,
    image: 'https://images.pexels.com/photos/996329/pexels-photo-996329.jpeg?auto=compress&cs=tinysrgb&w=400',
    rating: 4.3,
    reviews: 445,
    deal: {
      type: 'regular' as const,
      discount: 15,
      originalPrice: 10.58
    }
  },
  {
    id: '12',
    name: 'Home Décor Lamp',
    category: 'home',
    price: 34.99,
    image: 'https://images.pexels.com/photos/1884581/pexels-photo-1884581.jpeg?auto=compress&cs=tinysrgb&w=400',
    rating: 4.6,
    reviews: 267
  },
  {
    id: '13',
    name: 'Kitchen Blender',
    category: 'home',
    price: 89.99,
    image: 'https://images.pexels.com/photos/4226882/pexels-photo-4226882.jpeg?auto=compress&cs=tinysrgb&w=400',
    rating: 4.5,
    reviews: 678
  },
  {
    id: '14',
    name: 'Running Shoes',
    category: 'clothing',
    price: 79.99,
    image: 'https://images.pexels.com/photos/2529148/pexels-photo-2529148.jpeg?auto=compress&cs=tinysrgb&w=400',
    rating: 4.4,
    reviews: 523
  },
  {
    id: '15',
    name: 'Laptop Computer',
    category: 'electronics',
    price: 899.99,
    image: 'https://images.pexels.com/photos/205421/pexels-photo-205421.jpeg?auto=compress&cs=tinysrgb&w=400',
    rating: 4.7,
    reviews: 345
  },
  {
    id: '16',
    name: 'Coffee Maker',
    category: 'home',
    price: 129.99,
    image: 'https://images.pexels.com/photos/324028/pexels-photo-324028.jpeg?auto=compress&cs=tinysrgb&w=400',
    rating: 4.6,
    reviews: 234
  }
];

export const mockStats = {
  wasteReduced: 23,
  avgHoursToExpiry: 18,
  ngoPartners: 15
};

export const mockWasteData = [
  { date: 'Mon', wasteReduced: 120, predicted: 115 },
  { date: 'Tue', wasteReduced: 150, predicted: 145 },
  { date: 'Wed', wasteReduced: 180, predicted: 175 },
  { date: 'Thu', wasteReduced: 165, predicted: 170 },
  { date: 'Fri', wasteReduced: 200, predicted: 195 },
  { date: 'Sat', wasteReduced: 240, predicted: 235 },
  { date: 'Sun', wasteReduced: 220, predicted: 225 }
];

export const mockInventoryData = [
  { name: 'Low Risk', value: 65, color: '#00a651' },
  { name: 'Medium Risk', value: 25, color: '#ffc220' },
  { name: 'High Risk', value: 8, color: '#ff6b35' },
  { name: 'Critical', value: 2, color: '#e60012' }
];

export const mockNGOs = [
  {
    name: 'Food Bank Plus',
    distance: 2.3,
    deliveries: 28,
    capacity: 'High',
    contact: '+1 (555) 123-4567'
  },
  {
    name: 'Community Kitchen',
    distance: 4.1,
    deliveries: 15,
    capacity: 'Medium',
    contact: '+1 (555) 987-6543'
  },
  {
    name: 'Homeless Shelter Network',
    distance: 1.8,
    deliveries: 34,
    capacity: 'High',
    contact: '+1 (555) 456-7890'
  },
  {
    name: 'Senior Center Alliance',
    distance: 3.6,
    deliveries: 12,
    capacity: 'Low',
    contact: '+1 (555) 321-0987'
  }
];

export const mockPredictions = [
  {
    product: 'Organic Strawberries',
    quantity: 45,
    location: 'Produce Section A',
    expiryHours: 6,
    risk: 'high' as const,
    confidence: 94,
    suggestedAction: 'Apply 50% discount immediately'
  },
  {
    product: 'Whole Wheat Bread',
    quantity: 23,
    location: 'Bakery Section',
    expiryHours: 12,
    risk: 'medium' as const,
    confidence: 87,
    suggestedAction: 'Schedule NGO pickup'
  },
  {
    product: 'Greek Yogurt Cups',
    quantity: 67,
    location: 'Dairy Section B',
    expiryHours: 18,
    risk: 'medium' as const,
    confidence: 91,
    suggestedAction: 'Move to high-traffic area'
  },
  {
    product: 'Pre-made Salads',
    quantity: 34,
    location: 'Deli Counter',
    expiryHours: 4,
    risk: 'high' as const,
    confidence: 96,
    suggestedAction: 'Urgent discount + NGO alert'
  },
  {
    product: 'Rotisserie Chicken',
    quantity: 12,
    location: 'Hot Food Section',
    expiryHours: 24,
    risk: 'low' as const,
    confidence: 78,
    suggestedAction: 'Monitor closely'
  }
];