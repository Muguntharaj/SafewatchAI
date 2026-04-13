import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { motion } from 'motion/react';
import { Shield, Eye, Brain, Zap, Users, Lock, Globe, ExternalLink } from 'lucide-react';
import { Button } from './ui/button';
import neelamindsLogo from './assets/e8adf67d14cd54f8a5359d28bfc7cce80902b57e.png';

export function AboutPage() {
  const features = [
    {
      icon: <Eye className="w-8 h-8 text-blue-600" />,
      emoji: '👁️',
      title: 'Real-Time Monitoring',
      description: 'Monitor multiple CCTV cameras simultaneously with AI-powered detection'
    },
    {
      icon: <Brain className="w-8 h-8 text-purple-600" />,
      emoji: '🧠',
      title: 'Smart AI Detection',
      description: 'Advanced artificial intelligence for accurate person and activity detection'
    },
    {
      icon: <Users className="w-8 h-8 text-green-600" />,
      emoji: '👥',
      title: 'Face Recognition',
      description: 'Automated person classification with face recognition technology'
    },
    {
      icon: <Zap className="w-8 h-8 text-yellow-600" />,
      emoji: '⚡',
      title: 'Instant Alerts',
      description: 'Automatic email, SMS, and WhatsApp notifications for suspicious activities'
    },
    {
      icon: <Lock className="w-8 h-8 text-red-600" />,
      emoji: '🔒',
      title: 'Secure & Reliable',
      description: 'Enterprise-grade security with 24/7 continuous monitoring'
    },
    {
      icon: <Globe className="w-8 h-8 text-indigo-600" />,
      emoji: '🌐',
      title: 'Cloud-Based',
      description: 'Access your security system from anywhere, anytime'
    }
  ];

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* Hero Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center"
      >
        <div className="inline-block mb-4">
          <div className="w-24 h-24 mx-auto rounded-full bg-gradient-to-br from-blue-600 to-purple-600 flex items-center justify-center shadow-2xl">
            <Shield className="w-12 h-12 text-white" />
          </div>
        </div>
        <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-4">
          👁️ SafeWatch AI
        </h1>
        <p className="text-2xl font-semibold text-gray-700 mb-2">
          Smart Security, Smarter Safety
        </p>
        <p className="text-lg text-gray-600 max-w-3xl mx-auto">
          Your advanced AI-powered CCTV surveillance system for comprehensive security monitoring,
          intelligent person detection, and real-time threat alerts.
        </p>
      </motion.div>

      {/* Main Description Card */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <Card className="shadow-2xl border-2 border-blue-200 bg-gradient-to-br from-white to-blue-50">
          <CardHeader className="bg-gradient-to-r from-blue-600 to-purple-600 text-white">
            <CardTitle className="text-2xl">📖 About SafeWatch AI</CardTitle>
          </CardHeader>
          <CardContent className="pt-6 space-y-4 text-base leading-relaxed">
            <p className="text-gray-700">
              <strong className="text-blue-600">SafeWatch AI</strong> is a cutting-edge security surveillance platform that combines 
              the power of artificial intelligence with modern CCTV technology to provide unparalleled security monitoring 
              for businesses, institutions, and properties.
            </p>
            
            <p className="text-gray-700">
              Our system automatically detects and classifies persons entering your premises, distinguishing between 
              employees, owners, and unknown individuals. Using advanced <strong>face recognition technology</strong>, 
              SafeWatch AI remembers previously classified persons, eliminating repeated "unknown person" alerts and 
              streamlining your security operations.
            </p>

            <p className="text-gray-700">
              When suspicious activities are detected—such as loitering, trespassing, fighting, vandalism, theft, or 
              weapons—the system immediately sends automatic alerts via <strong>email, SMS, and WhatsApp</strong>, 
              complete with video clips of the incident. No popups, no delays—just instant, actionable intelligence.
            </p>

            <p className="text-gray-700">
              With <strong>automatic time tracking</strong>, SafeWatch AI logs in/out times for all detected persons, 
              providing comprehensive attendance and movement records. The intuitive dashboard features large emojis, 
              color-coded sections, and clickable stats cards designed for ease of use—even for users with limited 
              technical knowledge.
            </p>

            <div className="bg-blue-100 border-l-4 border-blue-600 p-4 rounded">
              <p className="text-blue-900 font-semibold">
                🎯 <strong>Mission:</strong> To make advanced security technology accessible, affordable, and easy to use 
                for everyone, ensuring safer environments through intelligent automation.
              </p>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Features Grid */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <h2 className="text-3xl font-bold text-center mb-6 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
          ✨ Key Features
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.3 + index * 0.1 }}
            >
              <Card className="shadow-lg hover:shadow-2xl transition-shadow h-full border-2 border-gray-200 hover:border-blue-300">
                <CardContent className="pt-6">
                  <div className="flex items-start gap-4">
                    <div className="flex-shrink-0">
                      <div className="text-4xl mb-2">{feature.emoji}</div>
                    </div>
                    <div>
                      <h3 className="font-bold text-lg mb-2 text-gray-800">{feature.title}</h3>
                      <p className="text-gray-600 text-sm">{feature.description}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Technology Stack */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
      >
        <Card className="shadow-xl border-2 border-purple-200">
          <CardHeader className="bg-gradient-to-r from-purple-50 to-pink-50">
            <CardTitle className="text-2xl text-center">⚙️ Technology Stack</CardTitle>
          </CardHeader>
          <CardContent className="pt-6">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
              <div className="p-4 bg-blue-50 rounded-lg">
                <div className="text-3xl mb-2">🤖</div>
                <div className="font-semibold text-sm">AI/ML Models</div>
                <div className="text-xs text-gray-600">Deep Learning</div>
              </div>
              <div className="p-4 bg-green-50 rounded-lg">
                <div className="text-3xl mb-2">📹</div>
                <div className="font-semibold text-sm">CCTV Integration</div>
                <div className="text-xs text-gray-600">Multi-Camera</div>
              </div>
              <div className="p-4 bg-purple-50 rounded-lg">
                <div className="text-3xl mb-2">👤</div>
                <div className="font-semibold text-sm">Face Recognition</div>
                <div className="text-xs text-gray-600">Biometric ID</div>
              </div>
              <div className="p-4 bg-orange-50 rounded-lg">
                <div className="text-3xl mb-2">📱</div>
                <div className="font-semibold text-sm">Multi-Channel Alerts</div>
                <div className="text-xs text-gray-600">SMS/Email/WhatsApp</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Neelaminds Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
      >
        <Card className="shadow-2xl border-4 border-blue-300 bg-gradient-to-br from-blue-50 to-purple-50">
          <CardContent className="pt-8 pb-8">
            <div className="text-center space-y-6">
              <div className="flex items-center justify-center gap-4">
                <img 
                  src={neelamindsLogo} 
                  alt="Neelaminds" 
                  className="w-32 h-32 object-contain drop-shadow-xl"
                />
                <div className="text-left">
                  <h2 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                    Neelaminds
                  </h2>
                  {/* <p className="text-lg text-gray-600 mt-1">Private Limited</p> */}
                </div>
              </div>

              <div className="max-w-3xl mx-auto space-y-4">
                <p className="text-lg text-gray-700 leading-relaxed">
                  <strong className="text-blue-600">Neelaminds Private Limited</strong> is a pioneering technology company 
                  specializing in artificial intelligence, machine learning, and smart automation solutions. We are dedicated 
                  to creating innovative products that solve real-world problems and make advanced technology accessible to everyone.
                </p>

                <p className="text-base text-gray-600">
                  With a focus on security, efficiency, and user experience, Neelaminds delivers enterprise-grade solutions 
                  that empower businesses to operate smarter and safer. Our team of expert engineers and data scientists 
                  work tirelessly to push the boundaries of what's possible with AI technology.
                </p>

                <div className="bg-gradient-to-r from-blue-100 to-purple-100 p-6 rounded-xl border-2 border-blue-300">
                  <p className="text-lg font-semibold text-gray-800 mb-4">
                    🇮🇳 <strong>Proudly Made in India</strong> with passion, innovation, and excellence
                  </p>
                  <div className="flex justify-center gap-3 text-sm">
                    <span className="bg-white px-4 py-2 rounded-full font-semibold text-blue-600">Innovation</span>
                    <span className="bg-white px-4 py-2 rounded-full font-semibold text-purple-600">Security</span>
                    <span className="bg-white px-4 py-2 rounded-full font-semibold text-green-600">Excellence</span>
                  </div>
                </div>

                <Button
                  size="lg"
                  className="mt-6 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white text-lg px-8 py-6 h-auto"
                  onClick={() => window.open('https://www.neelaminds.com', '_blank')}
                >
                  <Globe className="w-5 h-5 mr-2" />
                  🌐 Visit Neelaminds.com
                  <ExternalLink className="w-5 h-5 ml-2" />
                </Button>

                <p className="text-sm text-gray-500 mt-4">
                  Learn more about our products, services, and the innovative solutions we're building for tomorrow.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Contact & Support */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
        className="text-center pb-8"
      >
        <Card className="shadow-lg bg-gradient-to-r from-gray-50 to-blue-50">
          <CardContent className="py-6">
            <h3 className="text-xl font-bold text-gray-800 mb-3">📞 Need Support?</h3>
            <p className="text-gray-600 mb-4">
              Our team is here to help you 24/7 with any questions or technical assistance.
            </p>
            <div className="flex justify-center gap-4 flex-wrap">
              <span className="bg-white px-4 py-2 rounded-lg shadow text-sm">
                📧 <strong>Email:</strong> support@neelaminds.com
              </span>
              <span className="bg-white px-4 py-2 rounded-lg shadow text-sm">
                📱 <strong>Phone:</strong> +91-XXXX-XXXXXX
              </span>
              <span className="bg-white px-4 py-2 rounded-lg shadow text-sm">
                💬 <strong>WhatsApp:</strong> Quick Support
              </span>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Footer */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.7 }}
        className="text-center text-sm text-gray-500 pb-4"
      >
        <p>© 2026 Neelaminds Private Limited. All rights reserved.</p>
        <p className="mt-1">SafeWatch AI - Protecting what matters most 🛡️</p>
      </motion.div>
    </div>
  );
}
