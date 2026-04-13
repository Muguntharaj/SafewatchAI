import { Card, CardContent } from './ui/card';
import { Users, UserCheck, UserX, Activity, AlertTriangle } from 'lucide-react';
import { motion } from 'motion/react';

interface StatsCardsProps {
  stats: {
    total: number;
    known: number;
    unknown: number;
    currentlyIn: number;
    alerts: number;
  };
  onCardClick: (type: 'total' | 'known' | 'unknown' | 'active' | 'alerts') => void;
}

export function StatsCards({ stats, onCardClick }: StatsCardsProps) {
  const cards = [
    {
      id: 'total' as const,
      title: 'Total Visitors',
      subtitle: 'Everyone Detected',
      value: stats.total,
      icon: Users,
      gradient: 'from-blue-500 to-blue-600',
      emoji: '👥',
      hoverGradient: 'hover:from-blue-600 hover:to-blue-700'
    },
    {
      id: 'known' as const,
      title: 'Recognized Faces',
      subtitle: 'Trusted People',
      value: stats.known,
      icon: UserCheck,
      gradient: 'from-green-500 to-green-600',
      emoji: '✅',
      hoverGradient: 'hover:from-green-600 hover:to-green-700'
    },
    {
      id: 'unknown' as const,
      title: 'Mystery Visitors',
      subtitle: 'Need Your Attention',
      value: stats.unknown,
      icon: UserX,
      gradient: 'from-orange-500 to-orange-600',
      emoji: '❓',
      hoverGradient: 'hover:from-orange-600 hover:to-orange-700'
    },
    {
      id: 'active' as const,
      title: 'Live Presence',
      subtitle: 'Currently Inside',
      value: stats.currentlyIn,
      icon: Activity,
      gradient: 'from-purple-500 to-purple-600',
      emoji: '🚶',
      hoverGradient: 'hover:from-purple-600 hover:to-purple-700'
    },
    {
      id: 'alerts' as const,
      title: 'Threat Alerts',
      subtitle: 'Action Required',
      value: stats.alerts,
      icon: AlertTriangle,
      gradient: 'from-red-500 to-red-600',
      emoji: '⚠️',
      hoverGradient: 'hover:from-red-600 hover:to-red-700'
    }
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
      {cards.map((card, index) => (
        <motion.div
          key={card.id}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.1 }}
          whileHover={{ scale: 1.05, y: -8 }}
          whileTap={{ scale: 0.98 }}
        >
          <Card 
            className={`bg-gradient-to-br ${card.gradient} ${card.hoverGradient} text-white border-0 shadow-lg overflow-hidden relative cursor-pointer transition-all duration-300`}
            onClick={() => onCardClick(card.id)}
          >
            <CardContent className="p-6">
              {/* Background Icon */}
              <div className="absolute -right-4 -top-4 opacity-20">
                <card.icon className="w-24 h-24" />
              </div>

              {/* Content */}
              <div className="relative z-10">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-4xl">{card.emoji}</span>
                  <card.icon className="w-6 h-6 opacity-80" />
                </div>
                
                {/* Text bigger, number smaller */}
                <div className="mb-3">
                  <div className="text-xl font-bold leading-tight">{card.title}</div>
                  <div className="text-sm opacity-90 mt-0.5">{card.subtitle}</div>
                </div>
                
                <div className="text-3xl font-bold">
                  {card.value}
                </div>
              </div>

              {/* Click indicator */}
              <div className="absolute bottom-2 right-2 opacity-60">
                <span className="text-xs">Click to view →</span>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      ))}
    </div>
  );
}