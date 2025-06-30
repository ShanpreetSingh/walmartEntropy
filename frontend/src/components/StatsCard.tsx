import React from 'react';
import { DivideIcon as LucideIcon } from 'lucide-react';

interface StatsCardProps {
  icon: LucideIcon;
  title: string;
  value: string | number;
  suffix?: string;
  trend?: 'up' | 'down';
  bgColor?: string;
}

const StatsCard: React.FC<StatsCardProps> = ({ 
  icon: Icon, 
  title, 
  value, 
  suffix = '', 
  trend,
  bgColor = 'bg-blue-500' 
}) => {
  return (
    <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4 border border-white/20">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-blue-100 text-sm font-medium">{title}</p>
          <p className="text-2xl font-bold text-white">
            {value}{suffix}
          </p>
        </div>
        <div className={`${bgColor} p-3 rounded-lg`}>
          <Icon className="w-6 h-6 text-white" />
        </div>
      </div>
      {trend && (
        <div className="mt-2">
          <span className={`text-sm ${trend === 'up' ? 'text-green-300' : 'text-red-300'}`}>
            {trend === 'up' ? '↗' : '↘'} vs last week
          </span>
        </div>
      )}
    </div>
  );
};

export default StatsCard;