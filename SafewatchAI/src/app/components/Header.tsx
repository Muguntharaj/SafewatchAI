import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Shield, Bell, LogOut, Clock, Home, Info } from 'lucide-react';
import { motion } from 'motion/react';
import { useState, useEffect } from 'react';
import neelamindsLogo from './assets/e8adf67d14cd54f8a5359d28bfc7cce80902b57e.png';

interface HeaderProps {
  username: string;
  role: string;
  notificationCount: number;
  onNotificationClick: () => void;
  onLogout: () => void;
  currentView: 'home' | 'about';
  onViewChange: (view: 'home' | 'about') => void;
}

export function Header({ username, role, notificationCount, onNotificationClick, onLogout, currentView, onViewChange }: HeaderProps) {
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  return (
    <header className="bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 text-white px-6 py-4 shadow-lg">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          {/* Logo */}
          <motion.div
            whileHover={{ scale: 1.1, rotate: 5 }}
            className="w-12 h-12 rounded-full bg-white/20 backdrop-blur-sm flex items-center justify-center"
          >
            <Shield className="w-7 h-7 text-white" />
          </motion.div>
          
          {/* Title with Home and About Tabs */}
          <div>
            <h1 className="text-2xl flex items-center gap-2">
              👁️ <span className="font-bold">SafeWatch AI</span>
            </h1>
            <div className="flex items-center gap-2 mt-1">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => onViewChange('home')}
                className={`h-7 px-3 text-xs ${
                  currentView === 'home'
                    ? 'bg-white/30 text-white font-semibold'
                    : 'bg-white/10 text-blue-100 hover:bg-white/20 hover:text-white'
                }`}
              >
                <Home className="w-3 h-3 mr-1" />
                🏠 Home
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => onViewChange('about')}
                className={`h-7 px-3 text-xs ${
                  currentView === 'about'
                    ? 'bg-white/30 text-white font-semibold'
                    : 'bg-white/10 text-blue-100 hover:bg-white/20 hover:text-white'
                }`}
              >
                <Info className="w-3 h-3 mr-1" />
                ℹ️ About
              </Button>
            </div>
          </div>
        </div>

        {/* Right Section - Neelaminds Logo, Username and Actions in a row */}
        <div className="flex items-center gap-4">
          {/* Neelaminds Logo - Bigger and more visible */}
          <div className="flex items-center gap-1">
            <img 
              src={neelamindsLogo} 
              alt="Neelaminds" 
              className="w-32 h-32 object-contain drop-shadow-lg"
            />
            <span className="font-bold text-2xl">Neelaminds</span>
          </div>

          {/* Vertical Divider */}
          <div className="h-12 w-px bg-white/30" />

          {/* User Info - No role badge */}
          <div className="text-right">
            <div className="font-semibold text-base">{username}</div>
            <div className="text-xs text-blue-100">Active Session</div>
          </div>

          {/* Time Display */}
          <div className="hidden lg:flex items-center gap-2 bg-white/10 px-3 py-2 rounded-lg backdrop-blur-sm border border-white/20">
            <Clock className="w-4 h-4" />
            <span className="font-mono text-sm">
              {currentTime.toLocaleTimeString()}
            </span>
          </div>

          {/* Notifications Button */}
          <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
            <Button
              onClick={onNotificationClick}
              className="bg-white/20 hover:bg-white/30 backdrop-blur-sm h-11 px-4 relative border-2 border-white/30"
            >
              <Bell className="w-5 h-5" />
              {notificationCount > 0 && (
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  className="absolute -top-1 -right-1"
                >
                  <Badge className="bg-red-500 text-white h-5 w-5 p-0 flex items-center justify-center text-xs font-bold rounded-full">
                    {notificationCount}
                  </Badge>
                </motion.div>
              )}
            </Button>
          </motion.div>

          {/* Logout Button */}
          <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
            <Button
              variant="outline"
              onClick={onLogout}
              className="bg-white/10 hover:bg-red-500 hover:text-white border-2 border-white/30 h-11 px-4"
            >
              <LogOut className="w-5 h-5 mr-2" />
              <span className="hidden sm:inline">Logout</span>
            </Button>
          </motion.div>
        </div>
      </div>

      {/* Branding Bar */}
      <div className="mt-3 pt-3 border-t border-white/20 text-center">
        <p className="text-xs text-blue-100">
          Powered by <span className="font-bold text-white">Neelaminds</span> 🇮🇳 | Innovation • Security • Excellence
        </p>
      </div>
    </header>
  );
}