import React, { useState, useRef, useEffect } from 'react';
import { MessageCircle, X, Send, Bot, User, ShoppingCart, Sparkles, Lightbulb, Mic, MicOff, Image, Paperclip, Star, Heart, Share2, Copy, ThumbsUp, ThumbsDown } from 'lucide-react';
import { ChatbotService, ChatResponse } from '../services/chatbotService';

interface Message {
  id: string;
  type: 'user' | 'bot';
  content: string;
  timestamp: Date;
  products?: Array<{
    id: string;
    name: string;
    price: number;
    image: string;
    category: string;
    deal?: {
      discount: number;
      originalPrice: number;
    };
  }>;
  suggestions?: string[];
  isTyping?: boolean;
  feedback?: 'positive' | 'negative' | null;
}

const ChatBot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      type: 'bot',
      content: "Hi! I'm your Walmart AI assistant powered by SmartWaste technology. I can help you find the best deals, recommend products, and show you eco-friendly options that help reduce waste. What are you looking for today?",
      timestamp: new Date(),
      suggestions: ["Show me today's deals", "I need groceries", "Budget-friendly items", "Eco-friendly products"]
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [conversationContext, setConversationContext] = useState<string[]>([]);
  const [userPreferences, setUserPreferences] = useState({
    budget: '',
    categories: [] as string[],
    dietaryRestrictions: [] as string[]
  });
  const [showQuickActions, setShowQuickActions] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const chatbotService = ChatbotService.getInstance();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  const handleSendMessage = (messageText?: string) => {
    const textToSend = messageText || inputValue;
    if (!textToSend.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: textToSend,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsTyping(true);
    setShowQuickActions(false);

    // Update conversation context
    setConversationContext(prev => [...prev.slice(-4), textToSend]);

    // Show typing indicator immediately
    const typingMessage: Message = {
      id: `typing_${Date.now()}`,
      type: 'bot',
      content: '',
      timestamp: new Date(),
      isTyping: true
    };
    setMessages(prev => [...prev, typingMessage]);

    // Simulate realistic AI thinking time
    setTimeout(() => {
      const response: ChatResponse = chatbotService.generateResponse(textToSend, conversationContext, userPreferences);
      
      // Remove typing indicator and add actual response
      setMessages(prev => {
        const withoutTyping = prev.filter(m => !m.isTyping);
        const botMessage: Message = {
          id: (Date.now() + 1).toString(),
          type: 'bot',
          content: response.message,
          timestamp: new Date(),
          products: response.products,
          suggestions: response.suggestions
        };
        return [...withoutTyping, botMessage];
      });
      setIsTyping(false);
    }, Math.random() * 1500 + 1000);
  };

  const handleSuggestionClick = (suggestion: string) => {
    handleSendMessage(suggestion);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleProductAction = (productId: string, action: string) => {
    const actionMessage = `${action} product ${productId}`;
    handleSendMessage(actionMessage);
  };

  const handleVoiceInput = () => {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      alert('Speech recognition is not supported in your browser');
      return;
    }

    const SpeechRecognition = (window as any).webkitSpeechRecognition || (window as any).SpeechRecognition;
    const recognition = new SpeechRecognition();
    
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
      setIsListening(true);
    };

    recognition.onresult = (event: any) => {
      const transcript = event.results[0][0].transcript;
      setInputValue(transcript);
      setIsListening(false);
    };

    recognition.onerror = () => {
      setIsListening(false);
      alert('Speech recognition error. Please try again.');
    };

    recognition.onend = () => {
      setIsListening(false);
    };

    recognition.start();
  };

  const handleMessageFeedback = (messageId: string, feedback: 'positive' | 'negative') => {
    setMessages(prev => prev.map(msg => 
      msg.id === messageId ? { ...msg, feedback } : msg
    ));
    
    // Send feedback to service for learning
    chatbotService.recordFeedback(messageId, feedback);
  };

  const copyMessageToClipboard = (content: string) => {
    navigator.clipboard.writeText(content);
    // Could add a toast notification here
  };

  const quickActions = [
    { label: "Today's Deals", icon: "🔥", action: "Show me today's best deals" },
    { label: "Groceries", icon: "🛒", action: "I need groceries" },
    { label: "Electronics", icon: "📱", action: "Show me electronics" },
    { label: "Budget Items", icon: "💰", action: "Show me budget-friendly items" },
    { label: "Eco Deals", icon: "🌱", action: "Show me eco-friendly deals" },
    { label: "Flash Sales", icon: "⚡", action: "Any flash sales right now?" }
  ];

  return (
    <>
      {/* Chat Button */}
      <button
        onClick={() => setIsOpen(true)}
        className={`fixed bottom-6 right-6 bg-gradient-to-r from-blue-600 to-blue-700 text-white p-4 rounded-full shadow-lg hover:shadow-xl transition-all duration-300 z-50 ${
          isOpen ? 'hidden' : 'flex'
        } items-center justify-center group hover:scale-105`}
      >
        <MessageCircle className="w-6 h-6 group-hover:scale-110 transition-transform duration-200" />
        <div className="absolute -top-2 -right-2 bg-gradient-to-r from-green-400 to-green-500 text-white text-xs rounded-full w-6 h-6 flex items-center justify-center animate-pulse font-bold">
          AI
        </div>
        <div className="absolute -bottom-1 -right-1 w-3 h-3 bg-green-400 rounded-full animate-ping"></div>
      </button>

      {/* Chat Window */}
      {isOpen && (
        <div className="fixed bottom-6 right-6 w-96 h-[700px] bg-white rounded-2xl shadow-2xl border border-gray-200 flex flex-col z-50 overflow-hidden">
          {/* Header */}
          <div className="bg-gradient-to-r from-blue-600 to-blue-700 text-white p-4 flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-white/20 rounded-full flex items-center justify-center">
                <Sparkles className="w-5 h-5" />
              </div>
              <div>
                <h3 className="font-semibold">Walmart AI Assistant</h3>
                <p className="text-xs text-blue-100 flex items-center space-x-1">
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                  <span>SmartWaste AI Powered</span>
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <button
                onClick={() => setShowQuickActions(!showQuickActions)}
                className="text-white/80 hover:text-white transition-colors duration-200 hover:bg-white/10 p-1 rounded-full"
                title="Quick Actions"
              >
                <Lightbulb className="w-4 h-4" />
              </button>
              <button
                onClick={() => setIsOpen(false)}
                className="text-white/80 hover:text-white transition-colors duration-200 hover:bg-white/10 p-1 rounded-full"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* Quick Actions */}
          {showQuickActions && (
            <div className="bg-blue-50 p-3 border-b border-blue-100">
              <div className="grid grid-cols-3 gap-2">
                {quickActions.map((action, index) => (
                  <button
                    key={index}
                    onClick={() => handleSendMessage(action.action)}
                    className="flex flex-col items-center p-2 bg-white rounded-lg hover:bg-blue-100 transition-colors duration-200 text-xs"
                  >
                    <span className="text-lg mb-1">{action.icon}</span>
                    <span className="text-gray-700 text-center leading-tight">{action.label}</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div className={`flex items-start space-x-2 max-w-[85%] ${
                  message.type === 'user' ? 'flex-row-reverse space-x-reverse' : ''
                }`}>
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                    message.type === 'user' 
                      ? 'bg-blue-600 text-white' 
                      : 'bg-gradient-to-r from-green-400 to-green-500 text-white'
                  }`}>
                    {message.type === 'user' ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
                  </div>
                  
                  <div className={`p-3 rounded-2xl ${
                    message.type === 'user'
                      ? 'bg-blue-600 text-white'
                      : 'bg-white text-gray-800 shadow-sm border'
                  }`}>
                    {message.isTyping ? (
                      <div className="flex items-center space-x-2">
                        <div className="flex space-x-1">
                          <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                          <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                          <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                        </div>
                        <span className="text-xs text-gray-500">AI is thinking...</span>
                      </div>
                    ) : (
                      <>
                        <p className="text-sm leading-relaxed">{message.content}</p>
                        
                        {/* Product Recommendations */}
                        {message.products && message.products.length > 0 && (
                          <div className="mt-3 space-y-2">
                            {message.products.map((product) => (
                              <div key={product.id} className="flex items-center space-x-3 bg-gray-50 p-3 rounded-lg border hover:shadow-sm transition-shadow duration-200">
                                <img
                                  src={product.image}
                                  alt={product.name}
                                  className="w-12 h-12 object-cover rounded-lg"
                                />
                                <div className="flex-1 min-w-0">
                                  <p className="text-sm font-medium text-gray-800 truncate">{product.name}</p>
                                  <div className="flex items-center space-x-2">
                                    <p className="text-sm text-green-600 font-semibold">${product.price}</p>
                                    {product.deal && (
                                      <>
                                        <span className="text-xs text-gray-500 line-through">
                                          ${product.deal.originalPrice}
                                        </span>
                                        <span className="text-xs bg-red-100 text-red-600 px-1 py-0.5 rounded">
                                          {product.deal.discount}% OFF
                                        </span>
                                      </>
                                    )}
                                  </div>
                                  <p className="text-xs text-gray-500 capitalize">{product.category}</p>
                                </div>
                                <div className="flex flex-col space-y-1">
                                  <button 
                                    onClick={() => handleProductAction(product.id, 'Add to cart')}
                                    className="bg-blue-600 text-white p-2 rounded-full hover:bg-blue-700 transition-colors duration-200 flex-shrink-0"
                                    title="Add to Cart"
                                  >
                                    <ShoppingCart className="w-3 h-3" />
                                  </button>
                                  <button 
                                    onClick={() => handleProductAction(product.id, 'Add to wishlist')}
                                    className="bg-gray-100 text-gray-600 p-2 rounded-full hover:bg-gray-200 transition-colors duration-200 flex-shrink-0"
                                    title="Add to Wishlist"
                                  >
                                    <Heart className="w-3 h-3" />
                                  </button>
                                </div>
                              </div>
                            ))}
                          </div>
                        )}

                        {/* Suggestions */}
                        {message.suggestions && message.suggestions.length > 0 && (
                          <div className="mt-3 flex flex-wrap gap-2">
                            {message.suggestions.map((suggestion, index) => (
                              <button
                                key={index}
                                onClick={() => handleSuggestionClick(suggestion)}
                                className="text-xs bg-blue-50 text-blue-600 px-2 py-1 rounded-full hover:bg-blue-100 transition-colors duration-200 flex items-center space-x-1"
                              >
                                <Lightbulb className="w-3 h-3" />
                                <span>{suggestion}</span>
                              </button>
                            ))}
                          </div>
                        )}

                        {/* Message Actions (for bot messages) */}
                        {message.type === 'bot' && !message.isTyping && (
                          <div className="mt-2 flex items-center justify-between">
                            <div className="flex items-center space-x-2">
                              <button
                                onClick={() => handleMessageFeedback(message.id, 'positive')}
                                className={`p-1 rounded-full transition-colors duration-200 ${
                                  message.feedback === 'positive' 
                                    ? 'bg-green-100 text-green-600' 
                                    : 'text-gray-400 hover:text-green-600 hover:bg-green-50'
                                }`}
                                title="Helpful"
                              >
                                <ThumbsUp className="w-3 h-3" />
                              </button>
                              <button
                                onClick={() => handleMessageFeedback(message.id, 'negative')}
                                className={`p-1 rounded-full transition-colors duration-200 ${
                                  message.feedback === 'negative' 
                                    ? 'bg-red-100 text-red-600' 
                                    : 'text-gray-400 hover:text-red-600 hover:bg-red-50'
                                }`}
                                title="Not helpful"
                              >
                                <ThumbsDown className="w-3 h-3" />
                              </button>
                            </div>
                            <div className="flex items-center space-x-1">
                              <button
                                onClick={() => copyMessageToClipboard(message.content)}
                                className="p-1 rounded-full text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-colors duration-200"
                                title="Copy message"
                              >
                                <Copy className="w-3 h-3" />
                              </button>
                              <button
                                className="p-1 rounded-full text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-colors duration-200"
                                title="Share"
                              >
                                <Share2 className="w-3 h-3" />
                              </button>
                            </div>
                          </div>
                        )}
                      </>
                    )}
                  </div>
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="p-4 border-t border-gray-200 bg-white">
            <div className="flex items-center space-x-2 mb-2">
              <div className="flex-1 relative">
                <input
                  ref={inputRef}
                  type="text"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask about products, deals, or get recommendations..."
                  className="w-full p-3 pr-20 border border-gray-300 rounded-full focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm"
                  disabled={isTyping}
                />
                <div className="absolute right-2 top-1/2 transform -translate-y-1/2 flex items-center space-x-1">
                  <button
                    onClick={handleVoiceInput}
                    disabled={isTyping || isListening}
                    className={`p-2 rounded-full transition-colors duration-200 ${
                      isListening 
                        ? 'bg-red-100 text-red-600' 
                        : 'text-gray-400 hover:text-blue-600 hover:bg-blue-50'
                    }`}
                    title={isListening ? 'Listening...' : 'Voice input'}
                  >
                    {isListening ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
                  </button>
                </div>
              </div>
              <button
                onClick={() => handleSendMessage()}
                disabled={!inputValue.trim() || isTyping}
                className="bg-blue-600 text-white p-3 rounded-full hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors duration-200 flex-shrink-0"
              >
                <Send className="w-4 h-4" />
              </button>
            </div>
            <p className="text-xs text-gray-500 text-center">
              Powered by SmartWaste AI • Helping reduce waste while saving money
            </p>
          </div>
        </div>
      )}
    </>
  );
};

export default ChatBot;