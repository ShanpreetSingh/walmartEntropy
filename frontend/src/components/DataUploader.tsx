import React, { useState, useRef } from 'react';
import { Upload, FileText, AlertCircle, CheckCircle, X, Plus } from 'lucide-react';

interface TrainingDataRow {
  productName: string;
  category: string;
  purchaseDate: string;
  expiryDate: string;
  initialStock: number;
  finalStock: number;
  wasteAmount: number;
  temperature: number;
  humidity: number;
  promotionActive: boolean;
  location: string;
}

interface DataUploaderProps {
  onDataUpload: (data: TrainingDataRow[]) => void;
}

const DataUploader: React.FC<DataUploaderProps> = ({ onDataUpload }) => {
  const [uploadedData, setUploadedData] = useState<TrainingDataRow[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'success' | 'error'>('idle');
  const [manualEntry, setManualEntry] = useState<TrainingDataRow>({
    productName: '',
    category: 'produce',
    purchaseDate: '',
    expiryDate: '',
    initialStock: 0,
    finalStock: 0,
    wasteAmount: 0,
    temperature: 70,
    humidity: 60,
    promotionActive: false,
    location: ''
  });
  const [showManualEntry, setShowManualEntry] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    setUploadStatus('idle');

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const text = e.target?.result as string;
        const rows = text.split('\n').slice(1); // Skip header row
        const parsedData: TrainingDataRow[] = [];

        rows.forEach((row, index) => {
          if (row.trim()) {
            const columns = row.split(',');
            if (columns.length >= 11) {
              parsedData.push({
                productName: columns[0]?.trim() || '',
                category: columns[1]?.trim() || 'produce',
                purchaseDate: columns[2]?.trim() || '',
                expiryDate: columns[3]?.trim() || '',
                initialStock: parseInt(columns[4]?.trim()) || 0,
                finalStock: parseInt(columns[5]?.trim()) || 0,
                wasteAmount: parseInt(columns[6]?.trim()) || 0,
                temperature: parseFloat(columns[7]?.trim()) || 70,
                humidity: parseFloat(columns[8]?.trim()) || 60,
                promotionActive: columns[9]?.trim().toLowerCase() === 'true',
                location: columns[10]?.trim() || ''
              });
            }
          }
        });

        setUploadedData(parsedData);
        onDataUpload(parsedData);
        setUploadStatus('success');
      } catch (error) {
        console.error('Error parsing CSV:', error);
        setUploadStatus('error');
      } finally {
        setIsUploading(false);
      }
    };

    reader.readAsText(file);
  };

  const handleManualSubmit = () => {
    if (manualEntry.productName && manualEntry.purchaseDate && manualEntry.expiryDate) {
      const newData = [...uploadedData, manualEntry];
      setUploadedData(newData);
      onDataUpload(newData);
      setManualEntry({
        productName: '',
        category: 'produce',
        purchaseDate: '',
        expiryDate: '',
        initialStock: 0,
        finalStock: 0,
        wasteAmount: 0,
        temperature: 70,
        humidity: 60,
        promotionActive: false,
        location: ''
      });
      setShowManualEntry(false);
    }
  };

  const removeDataRow = (index: number) => {
    const newData = uploadedData.filter((_, i) => i !== index);
    setUploadedData(newData);
    onDataUpload(newData);
  };

  const downloadTemplate = () => {
    const csvContent = `Product Name,Category,Purchase Date,Expiry Date,Initial Stock,Final Stock,Waste Amount,Temperature,Humidity,Promotion Active,Location
Organic Bananas,produce,2024-01-15,2024-01-22,50,42,8,68,65,false,Produce Section A
Fresh Milk,dairy,2024-01-10,2024-01-17,30,28,2,38,45,true,Dairy Section B
Premium Bread,bakery,2024-01-12,2024-01-15,20,15,5,72,55,false,Bakery Section`;

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'waste_training_template.csv';
    a.click();
    window.URL.revokeObjectURL(url);
  };

  return (
    <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">Upload Training Data</h3>
      
      {/* Upload Methods */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        {/* CSV Upload */}
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-blue-400 transition-colors duration-200">
          <Upload className="w-8 h-8 text-gray-400 mx-auto mb-3" />
          <h4 className="font-medium text-gray-800 mb-2">Upload CSV File</h4>
          <p className="text-sm text-gray-600 mb-4">
            Upload your historical waste data in CSV format
          </p>
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv"
            onChange={handleFileUpload}
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
            className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors duration-200 disabled:opacity-50"
          >
            {isUploading ? 'Uploading...' : 'Choose File'}
          </button>
          <div className="mt-2">
            <button
              onClick={downloadTemplate}
              className="text-sm text-blue-600 hover:text-blue-700 underline"
            >
              Download CSV Template
            </button>
          </div>
        </div>

        {/* Manual Entry */}
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-green-400 transition-colors duration-200">
          <Plus className="w-8 h-8 text-gray-400 mx-auto mb-3" />
          <h4 className="font-medium text-gray-800 mb-2">Manual Entry</h4>
          <p className="text-sm text-gray-600 mb-4">
            Add individual data points manually
          </p>
          <button
            onClick={() => setShowManualEntry(true)}
            className="bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 transition-colors duration-200"
          >
            Add Data Point
          </button>
        </div>
      </div>

      {/* Upload Status */}
      {uploadStatus !== 'idle' && (
        <div className={`p-4 rounded-lg mb-4 flex items-center space-x-2 ${
          uploadStatus === 'success' ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'
        }`}>
          {uploadStatus === 'success' ? (
            <CheckCircle className="w-5 h-5" />
          ) : (
            <AlertCircle className="w-5 h-5" />
          )}
          <span>
            {uploadStatus === 'success' 
              ? `Successfully uploaded ${uploadedData.length} data points`
              : 'Error uploading file. Please check the format and try again.'
            }
          </span>
        </div>
      )}

      {/* Manual Entry Form */}
      {showManualEntry && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white p-6 rounded-xl max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-4">
              <h4 className="text-lg font-semibold text-gray-800">Add Training Data Point</h4>
              <button
                onClick={() => setShowManualEntry(false)}
                className="text-gray-400 hover:text-gray-600"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Product Name</label>
                <input
                  type="text"
                  value={manualEntry.productName}
                  onChange={(e) => setManualEntry({...manualEntry, productName: e.target.value})}
                  className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  placeholder="e.g., Organic Bananas"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Category</label>
                <select
                  value={manualEntry.category}
                  onChange={(e) => setManualEntry({...manualEntry, category: e.target.value})}
                  className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                >
                  <option value="produce">Produce</option>
                  <option value="dairy">Dairy</option>
                  <option value="bakery">Bakery</option>
                  <option value="meat">Meat</option>
                  <option value="frozen">Frozen</option>
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Purchase Date</label>
                <input
                  type="date"
                  value={manualEntry.purchaseDate}
                  onChange={(e) => setManualEntry({...manualEntry, purchaseDate: e.target.value})}
                  className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Expiry Date</label>
                <input
                  type="date"
                  value={manualEntry.expiryDate}
                  onChange={(e) => setManualEntry({...manualEntry, expiryDate: e.target.value})}
                  className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Initial Stock</label>
                <input
                  type="number"
                  value={manualEntry.initialStock}
                  onChange={(e) => setManualEntry({...manualEntry, initialStock: parseInt(e.target.value) || 0})}
                  className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  placeholder="50"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Final Stock</label>
                <input
                  type="number"
                  value={manualEntry.finalStock}
                  onChange={(e) => setManualEntry({...manualEntry, finalStock: parseInt(e.target.value) || 0})}
                  className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  placeholder="42"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Waste Amount</label>
                <input
                  type="number"
                  value={manualEntry.wasteAmount}
                  onChange={(e) => setManualEntry({...manualEntry, wasteAmount: parseInt(e.target.value) || 0})}
                  className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  placeholder="8"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Temperature (°F)</label>
                <input
                  type="number"
                  value={manualEntry.temperature}
                  onChange={(e) => setManualEntry({...manualEntry, temperature: parseFloat(e.target.value) || 70})}
                  className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  placeholder="70"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Humidity (%)</label>
                <input
                  type="number"
                  value={manualEntry.humidity}
                  onChange={(e) => setManualEntry({...manualEntry, humidity: parseFloat(e.target.value) || 60})}
                  className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  placeholder="60"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Location</label>
                <input
                  type="text"
                  value={manualEntry.location}
                  onChange={(e) => setManualEntry({...manualEntry, location: e.target.value})}
                  className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  placeholder="Produce Section A"
                />
              </div>
              
              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="promotion"
                  checked={manualEntry.promotionActive}
                  onChange={(e) => setManualEntry({...manualEntry, promotionActive: e.target.checked})}
                  className="mr-2"
                />
                <label htmlFor="promotion" className="text-sm font-medium text-gray-700">
                  Promotion Active
                </label>
              </div>
            </div>
            
            <div className="flex justify-end space-x-3 mt-6">
              <button
                onClick={() => setShowManualEntry(false)}
                className="px-4 py-2 text-gray-600 border border-gray-300 rounded-lg hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={handleManualSubmit}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                Add Data Point
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Uploaded Data Preview */}
      {uploadedData.length > 0 && (
        <div className="mt-6">
          <h4 className="font-medium text-gray-800 mb-3">Uploaded Data ({uploadedData.length} records)</h4>
          <div className="max-h-64 overflow-y-auto border border-gray-200 rounded-lg">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 sticky top-0">
                <tr>
                  <th className="px-3 py-2 text-left">Product</th>
                  <th className="px-3 py-2 text-left">Category</th>
                  <th className="px-3 py-2 text-left">Stock</th>
                  <th className="px-3 py-2 text-left">Waste</th>
                  <th className="px-3 py-2 text-left">Actions</th>
                </tr>
              </thead>
              <tbody>
                {uploadedData.map((row, index) => (
                  <tr key={index} className="border-t border-gray-200">
                    <td className="px-3 py-2">{row.productName}</td>
                    <td className="px-3 py-2 capitalize">{row.category}</td>
                    <td className="px-3 py-2">{row.initialStock} → {row.finalStock}</td>
                    <td className="px-3 py-2">{row.wasteAmount}</td>
                    <td className="px-3 py-2">
                      <button
                        onClick={() => removeDataRow(index)}
                        className="text-red-600 hover:text-red-700"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default DataUploader;