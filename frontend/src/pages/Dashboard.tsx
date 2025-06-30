import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell } from 'recharts';
import { TrendingDown, AlertTriangle, Users, Package, Calendar, MapPin } from 'lucide-react';
import DashboardCard from '../components/DashboardCard';
import { mockWasteData, mockInventoryData, mockNGOs, mockPredictions } from '../data/mockData';

const Dashboard = () => {
  const [selectedTimeRange, setSelectedTimeRange] = useState('7d');

  const COLORS = ['#0071ce', '#ffc220', '#00a651', '#ff6b35'];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-800 mb-2">SmartWaste AI Dashboard</h1>
        <p className="text-gray-600">Predictive waste reduction and inventory management system</p>
      </div>

      {/* Time Range Selector */}
      <div className="mb-6">
        <div className="flex space-x-2">
          {['24h', '7d', '30d', '90d'].map((range) => (
            <button
              key={range}
              onClick={() => setSelectedTimeRange(range)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                selectedTimeRange === range
                  ? 'bg-blue-600 text-white shadow-md'
                  : 'bg-white text-gray-600 hover:bg-blue-50 border border-gray-200'
              }`}
            >
              {range}
            </button>
          ))}
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <DashboardCard
          title="Waste Prevented"
          value="2,847"
          unit="lbs"
          change="+23%"
          trend="up"
          icon={TrendingDown}
          color="green"
        />
        <DashboardCard
          title="At-Risk Items"
          value="156"
          unit="items"
          change="-12%"
          trend="down"
          icon={AlertTriangle}
          color="orange"
        />
        <DashboardCard
          title="NGO Donations"
          value="89"
          unit="deliveries"
          change="+45%"
          trend="up"
          icon={Users}
          color="purple"
        />
        <DashboardCard
          title="Revenue Saved"
          value="$12,450"
          unit=""
          change="+18%"
          trend="up"
          icon={Package}
          color="blue"
        />
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        {/* Waste Reduction Trend */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Waste Reduction Trend</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={mockWasteData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="wasteReduced" stroke="#0071ce" strokeWidth={3} />
              <Line type="monotone" dataKey="predicted" stroke="#ffc220" strokeWidth={2} strokeDasharray="5 5" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Inventory Risk Distribution */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Inventory Risk Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={mockInventoryData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {mockInventoryData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Predictions and Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        {/* AI Predictions */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">AI Predictions - Next 48 Hours</h3>
          <div className="space-y-4">
            {mockPredictions.map((prediction, index) => (
              <div key={index} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className={`w-3 h-3 rounded-full ${
                    prediction.risk === 'high' ? 'bg-red-500' :
                    prediction.risk === 'medium' ? 'bg-yellow-500' : 'bg-green-500'
                  }`}></div>
                  <div>
                    <p className="font-medium text-gray-800">{prediction.product}</p>
                    <p className="text-sm text-gray-600">{prediction.quantity} units • {prediction.location}</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-sm font-medium text-gray-800">{prediction.expiryHours}h left</p>
                  <p className="text-xs text-gray-500">{prediction.confidence}% confidence</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* NGO Partners */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">NGO Partnership Network</h3>
          <div className="space-y-4">
            {mockNGOs.map((ngo, index) => (
              <div key={index} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
                    <Users className="w-5 h-5 text-blue-600" />
                  </div>
                  <div>
                    <p className="font-medium text-gray-800">{ngo.name}</p>
                    <div className="flex items-center space-x-2 text-sm text-gray-600">
                      <MapPin className="w-3 h-3" />
                      <span>{ngo.distance} miles away</span>
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-sm font-medium text-green-600">{ngo.deliveries} deliveries</p>
                  <p className="text-xs text-gray-500">This month</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Action Center */}
      <div className="bg-gradient-to-r from-blue-600 to-blue-700 p-6 rounded-xl text-white">
        <h3 className="text-xl font-semibold mb-4">Recommended Actions</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-white/10 p-4 rounded-lg backdrop-blur-sm">
            <div className="flex items-center space-x-2 mb-2">
              <AlertTriangle className="w-5 h-5 text-yellow-400" />
              <span className="font-medium">Urgent Discount</span>
            </div>
            <p className="text-sm text-blue-100">Apply 40% discount to dairy products expiring in 12 hours</p>
          </div>
          <div className="bg-white/10 p-4 rounded-lg backdrop-blur-sm">
            <div className="flex items-center space-x-2 mb-2">
              <Users className="w-5 h-5 text-green-400" />
              <span className="font-medium">NGO Donation</span>
            </div>
            <p className="text-sm text-blue-100">Schedule pickup with Food Bank Plus for produce section</p>
          </div>
          <div className="bg-white/10 p-4 rounded-lg backdrop-blur-sm">
            <div className="flex items-center space-x-2 mb-2">
              <Package className="w-5 h-5 text-purple-400" />
              <span className="font-medium">Inventory Transfer</span>
            </div>
            <p className="text-sm text-blue-100">Move 50 units of bread to high-demand store location</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;