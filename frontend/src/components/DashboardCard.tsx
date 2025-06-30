import React from 'react';
import { DivideIcon as LucideIcon } from 'lucide-react';

interface DashboardCardProps {
  title: string;
  value: string;
  unit: string;
  change: string;
  trend: 'up' | 'down';
  icon: LucideIcon;
  color: 'blue' | 'green' | 'orange' | 'purple';
}

const DashboardCard: React.FC<DashboardCardProps> = ({
  title,
  value,
  unit,
  change,
  trend,
  icon: Icon,
  color
}) => {
  const colorClasses = {
    blue: 'bg-blue-50 text-blue-600 border-blue-100',
    green: 'bg-green-50 text-green-600 border-green-100',
    orange: 'bg-orange-50 text-orange-600 border-orange-100',
    purple: 'bg-purple-50 text-purple-600 border-purple-100'
  };

  const trendColor = trend === 'up' ? 'text-green-600' : 'text-red-600';

  return (
    <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-gray-600">{title}</h3>
        <div className={`p-2 rounded-lg ${colorClasses[color]}`}>
          <Icon className="w-5 h-5" />
        </div>
      </div>
      
      <div className="flex items-end justify-between">
        <div>
          <p className="text-2xl font-bold text-gray-800">
            {value} <span className="text-lg font-normal text-gray-500">{unit}</span>
          </p>
        </div>
        <div className={`text-sm font-medium ${trendColor}`}>
          {change}
        </div>
      </div>
    </div>
  );
};

export default DashboardCard;