import React from 'react';
import { Sparkles, TrendingDown, Clock, MapPin } from 'lucide-react';
import ProductCard from '../components/ProductCard';
import StatsCard from '../components/StatsCard';
import { mockProducts, mockStats } from '../data/mockData';

const Home = () => {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-blue-600 via-blue-700 to-blue-800 rounded-2xl p-8 mb-8 text-white relative overflow-hidden">
        <div className="absolute top-0 right-0 w-64 h-64 bg-yellow-400 rounded-full opacity-10 transform translate-x-32 -translate-y-32"></div>
        <div className="relative z-10">
          <div className="flex items-center space-x-2 mb-4">
            <Sparkles className="w-8 h-8 text-yellow-400" />
            <h1 className="text-3xl font-bold">Smart Deals & Waste Reduction</h1>
          </div>
          <p className="text-xl mb-6 text-blue-100">
            AI-powered platform reducing waste while offering amazing deals to customers
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <StatsCard
              icon={TrendingDown}
              title="Waste Reduced"
              value={mockStats.wasteReduced}
              suffix="%"
              trend="down"
              bgColor="bg-green-500"
            />
            <StatsCard
              icon={Clock}
              title="Hours to Expiry"
              value={mockStats.avgHoursToExpiry}
              suffix="hrs"
              bgColor="bg-orange-500"
            />
            <StatsCard
              icon={MapPin}
              title="NGO Partners"
              value={mockStats.ngoPartners}
              suffix="+"
              bgColor="bg-purple-500"
            />
          </div>
        </div>
      </div>

      {/* Flash Deals Section */}
      <section className="mb-12">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-gray-800">⚡ Flash Deals - Limited Time</h2>
          <span className="bg-red-100 text-red-600 px-3 py-1 rounded-full text-sm font-medium">
            Expires in 2 hours
          </span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {mockProducts.filter(p => p.deal?.type === 'flash').map(product => (
            <ProductCard key={product.id} product={product} />
          ))}
        </div>
      </section>

      {/* Smart Waste Deals */}
      <section className="mb-12">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-2xl font-bold text-gray-800">🌱 Smart Waste Reduction Deals</h2>
            <p className="text-gray-600">AI-predicted deals to prevent food waste</p>
          </div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {mockProducts.filter(p => p.deal?.type === 'waste-reduction').map(product => (
            <ProductCard key={product.id} product={product} />
          ))}
        </div>
      </section>

      {/* Regular Deals */}
      <section>
        <h2 className="text-2xl font-bold text-gray-800 mb-6">🛍️ All Deals</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {mockProducts.filter(p => !p.deal || (p.deal.type !== 'flash' && p.deal.type !== 'waste-reduction')).map(product => (
            <ProductCard key={product.id} product={product} />
          ))}
        </div>
      </section>
    </div>
  );
};

export default Home;