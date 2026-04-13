import { useState } from "react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./ui/card";
import { Shield, Eye, EyeOff, User, Lock } from "lucide-react";
import { motion } from "motion/react";
import backgroundImage from "./assets/f225d3f98a8b944ba8e5402ba79361cd00968af8.png";
import neelamindsLogo from "./assets/e8adf67d14cd54f8a5359d28bfc7cce80902b57e.png";

interface LoginPageProps {
  onLogin: (username: string, role: string) => void;
}

export function LoginPage({ onLogin }: LoginPageProps) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    // Mock authentication
    if (username && password) {
      // Simple mock role assignment
      const role = username.toLowerCase().includes("admin")
        ? "Admin"
        : "Security";
      onLogin(username, role);
    } else {
      setError("Please enter both username and password");
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center relative overflow-hidden">
      {/* Background Image with Lighter Overlay */}
      <div
        className="absolute inset-0 bg-cover bg-center"
        style={{
          backgroundImage: `url(${backgroundImage})`,
        }}
      >
        <div className="absolute inset-0 bg-black/40 backdrop-blur-[2px]" />
      </div>

      {/* Neelaminds Logo - Top Right */}
      <motion.div
        initial={{ opacity: 0, x: 50 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.2 }}
        className="absolute top-8 right-8 z-20 flex items-center gap-2"
      >
        <img
          src={neelamindsLogo}
          alt="Neelaminds"
          className="w-32 h-32 object-contain drop-shadow-2xl"
        />
        <span className="text-white font-bold text-3xl drop-shadow-lg">
          Neelaminds
        </span>
      </motion.div>

      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden">
        <motion.div
          className="absolute w-96 h-96 bg-blue-500/10 rounded-full blur-3xl"
          animate={{
            x: [0, 100, 0],
            y: [0, 50, 0],
          }}
          transition={{
            duration: 20,
            repeat: Infinity,
            ease: "linear",
          }}
          style={{ top: "10%", left: "10%" }}
        />
        <motion.div
          className="absolute w-96 h-96 bg-purple-500/10 rounded-full blur-3xl"
          animate={{
            x: [0, -100, 0],
            y: [0, -50, 0],
          }}
          transition={{
            duration: 15,
            repeat: Infinity,
            ease: "linear",
          }}
          style={{ bottom: "10%", right: "10%" }}
        />
      </div>

      <Card className="w-full max-w-md mx-4 shadow-2xl relative z-10 bg-white/95 backdrop-blur-md border-2">
        <CardHeader className="space-y-3 text-center pb-2">
          {/* Logo/Icon */}
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: "spring", duration: 0.8 }}
            className="mx-auto"
          >
            <div className="w-24 h-24 rounded-full bg-gradient-to-br from-blue-600 to-purple-600 flex items-center justify-center mb-2 shadow-lg">
              <Shield className="w-12 h-12 text-white" />
            </div>
          </motion.div>

          {/* Catchy Title */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <CardTitle className="text-3xl bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              👁️ SafeWatch AI
            </CardTitle>
            <p className="text-lg font-semibold text-slate-700 mt-1">
              Your Eyes Everywhere, Security Anywhere
            </p>
          </motion.div>

          <CardDescription className="text-base">
            🔒 Intelligent Surveillance at Your Fingertips
          </CardDescription>
        </CardHeader>

        <CardContent className="pt-4">
          <form onSubmit={handleSubmit} className="space-y-5">
            {/* Username Field */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 }}
              className="space-y-2"
            >
              <Label
                htmlFor="username"
                className="text-base flex items-center gap-2"
              >
                <User className="w-5 h-5 text-blue-600" />
                <span>Your Username</span>
              </Label>
              <Input
                id="username"
                type="text"
                placeholder="Enter your name"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="h-12 text-lg"
              />
            </motion.div>

            {/* Password Field */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.5 }}
              className="space-y-2"
            >
              <Label
                htmlFor="password"
                className="text-base flex items-center gap-2"
              >
                <Lock className="w-5 h-5 text-blue-600" />
                <span>Your Password</span>
              </Label>
              <div className="relative">
                <Input
                  id="password"
                  type={showPassword ? "text" : "password"}
                  placeholder="Enter password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="h-12 text-lg pr-12"
                />
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="absolute right-2 top-2"
                  onClick={() => setShowPassword(!showPassword)}
                >
                  {showPassword ? (
                    <EyeOff className="w-5 h-5" />
                  ) : (
                    <Eye className="w-5 h-5" />
                  )}
                </Button>
              </div>
            </motion.div>

            {error && (
              <motion.p
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="text-sm text-red-600 bg-red-50 p-3 rounded-lg border border-red-200"
              >
                ⚠️ {error}
              </motion.p>
            )}

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
            >
              <Button
                type="submit"
                className="w-full h-14 text-lg bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700"
              >
                🚀 Access Dashboard
              </Button>
            </motion.div>

            <div className="text-sm text-center text-gray-500 bg-blue-50 p-3 rounded-lg border border-blue-100">
              💡 <strong>Quick Start:</strong> Enter any
              username and password to begin
            </div>
          </form>

          {/* Branding */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.8 }}
            className="mt-6 pt-4 border-t text-center"
          >
            <p className="text-xs text-gray-500 mb-1">
              Powered by
            </p>
            <p className="font-bold text-lg bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Neelaminds
            </p>
            <p className="text-xs text-gray-500 mt-1">
              🇮🇳 Made in India with ❤️
            </p>
          </motion.div>
        </CardContent>
      </Card>
    </div>
  );
}