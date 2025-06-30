import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navigation from './components/Navigation';
import Home from './pages/Home';
import Dashboard from './pages/Dashboard';
import Products from './pages/Products';
import DataTraining from './pages/DataTraining';
import ChatBot from './components/ChatBot';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <Navigation />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/products" element={<Products />} />
          <Route path="/training" element={<DataTraining />} />
        </Routes>
        <ChatBot />
      </div>
    </Router>
  );
}

export default App;