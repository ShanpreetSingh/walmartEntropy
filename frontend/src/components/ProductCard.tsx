import React from 'react';
import { Clock, Zap, Leaf, ShoppingCart, Star } from 'lucide-react';

interface Deal {
  type: 'flash' | 'waste-reduction' | 'regular';
  discount: number;
  originalPrice: number;
  expiryHours?: number;
}

interface Product {
  id: string;
  name: string;
  category: string;
  price: number;
  image: string;
  rating: number;
  reviews: number;
  deal?: Deal;
  wasteRisk?: 'low' | 'medium' | 'high';
}

interface ProductCardProps {
  product: Product;
  viewMode?: 'grid' | 'list';
}

const ProductCard: React.FC<ProductCardProps> = ({ product, viewMode = 'grid' }) => {
  const getDealBadge = () => {
    if (!product.deal) return null;

    switch (product.deal.type) {
      case 'flash':
        return (
          <div className="absolute top-2 left-2 bg-red-500 text-white px-2 py-1 rounded-full text-xs font-bold flex items-center space-x-1">
            <Zap className="w-3 h-3" />
            <span>FLASH</span>
          </div>
        );
      case 'waste-reduction':
        return (
          <div className="absolute top-2 left-2 bg-green-500 text-white px-2 py-1 rounded-full text-xs font-bold flex items-center space-x-1">
            <Leaf className="w-3 h-3" />
            <span>ECO</span>
          </div>
        );
      default:
        return null;
    }
  };

  const getRiskIndicator = () => {
    if (!product.wasteRisk) return null;

    const colors = {
      low: 'bg-green-100 text-green-800',
      medium: 'bg-yellow-100 text-yellow-800',
      high: 'bg-red-100 text-red-800'
    };

    return (
      <div className={`absolute top-2 right-2 px-2 py-1 rounded-full text-xs font-medium ${colors[product.wasteRisk]}`}>
        {product.wasteRisk.toUpperCase()} RISK
      </div>
    );
  };

  if (viewMode === 'list') {
    return (
      <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-100 flex items-center space-x-4 hover:shadow-md transition-shadow duration-200">
        <div className="relative flex-shrink-0">
          <img
            src={product.image}
            alt={product.name}
            className="w-20 h-20 object-cover rounded-lg"
          />
          {getDealBadge()}
        </div>
        
        <div className="flex-1">
          <h3 className="font-semibold text-gray-800 mb-1">{product.name}</h3>
          <div className="flex items-center space-x-2 mb-2">
            <div className="flex items-center space-x-1">
              <Star className="w-4 h-4 text-yellow-500 fill-current" />
              <span className="text-sm text-gray-600">{product.rating}</span>
              <span className="text-sm text-gray-400">({product.reviews})</span>
            </div>
            <span className="text-sm text-gray-400 capitalize">{product.category}</span>
          </div>
          
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <span className="text-xl font-bold text-gray-800">${product.price}</span>
              {product.deal && (
                <span className="text-sm text-gray-500 line-through">
                  ${product.deal.originalPrice}
                </span>
              )}
              {product.deal && (
                <span className="text-sm font-medium text-green-600">
                  {product.deal.discount}% OFF
                </span>
              )}
            </div>
            
            <button className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors duration-200 flex items-center space-x-1">
              <ShoppingCart className="w-4 h-4" />
              <span>Add to Cart</span>
            </button>
          </div>
        </div>
        
        {getRiskIndicator()}
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden hover:shadow-lg transition-all duration-200 transform hover:-translate-y-1">
      <div className="relative">
        <img
          src={product.image}
          alt={product.name}
          className="w-full h-48 object-cover"
        />
        {getDealBadge()}
        {getRiskIndicator()}
        
        {product.deal?.expiryHours && (
          <div className="absolute bottom-2 left-2 bg-black/70 text-white px-2 py-1 rounded-full text-xs flex items-center space-x-1">
            <Clock className="w-3 h-3" />
            <span>{product.deal.expiryHours}h left</span>
          </div>
        )}
      </div>
      
      <div className="p-4">
        <h3 className="font-semibold text-gray-800 mb-2">{product.name}</h3>
        
        <div className="flex items-center space-x-2 mb-3">
          <div className="flex items-center space-x-1">
            <Star className="w-4 h-4 text-yellow-500 fill-current" />
            <span className="text-sm text-gray-600">{product.rating}</span>
            <span className="text-sm text-gray-400">({product.reviews})</span>
          </div>
        </div>
        
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-2">
            <span className="text-xl font-bold text-gray-800">${product.price}</span>
            {product.deal && (
              <>
                <span className="text-sm text-gray-500 line-through">
                  ${product.deal.originalPrice}
                </span>
                <span className="text-sm font-medium text-green-600">
                  {product.deal.discount}% OFF
                </span>
              </>
            )}
          </div>
        </div>
        
        <button className="w-full bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 transition-colors duration-200 flex items-center justify-center space-x-2">
          <ShoppingCart className="w-4 h-4" />
          <span>Add to Cart</span>
        </button>
      </div>
    </div>
  );
};

export default ProductCard;