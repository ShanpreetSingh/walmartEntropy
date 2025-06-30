import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { ShoppingCart, BarChart3, Package, Sparkles, Brain } from 'lucide-react';

const Navigation = () => {
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Deals & Offers', icon: ShoppingCart },
    { path: '/products', label: 'Products', icon: Package },
    { path: '/dashboard', label: 'SmartWaste AI', icon: BarChart3 },
    { path: '/training', label: 'AI Training', icon: Brain },
  ];

  return (
    <nav className="bg-white shadow-lg border-b-4 border-blue-600">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <Link to="/" className="flex items-center space-x-2">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-blue-700 rounded-lg flex items-center justify-center">
                <Sparkles className="w-6 h-6 text-yellow-400" />
              </div>
              <div className="flex flex-col">
                <span className="text-xl font-bold text-blue-600">Walmart</span>
                <span className="text-xs text-gray-500">SmartWaste AI</span>
              </div>
            </Link>
          </div>
          
          <div className="flex items-center space-x-8">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.path;
              
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium transition-all duration-200 ${
                    isActive
                      ? 'bg-blue-100 text-blue-700 border-b-2 border-blue-600'
                      : 'text-gray-600 hover:text-blue-600 hover:bg-blue-50'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{item.label}</span>
                </Link>
              );
            })}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;