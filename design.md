# Design Document: KrishiSevak Platform

## Overview

KrishiSevak is a comprehensive AI-powered agricultural platform designed to empower Indian farmers with accurate crop yield predictions and personalized farming recommendations. The system architecture follows a microservices pattern with clear separation between data collection, AI/ML processing, recommendation generation, and user interfaces.

### Current Development Status

**70% Completed** - Core platform functionality is operational with ongoing refinement of advanced features.

**Completed:**
- Frontend dashboard (React.js/Next.js PWA)
- User authentication and authorization
- Basic ML models (Random Forest, XGBoost)
- Database schema and data models
- Weather data integration
- Basic recommendation engine

**In Progress (20%):**
- Voice interface with 15+ regional languages
- Google Earth Engine and Agromonitoring satellite integration
- OpenAI API integration for conversational AI
- Offline sync functionality
- Multi-channel alert system

**Pending (10%):**
- Comprehensive testing and validation
- Performance optimization
- Production deployment
- Extension officer dashboard
- Government scheme integration

### Design Principles

1. **Accessibility First**: Design for low-literacy users with voice interfaces and visual communication
2. **Offline Resilience**: Core functionality available without internet connectivity
3. **Performance on Constraints**: Optimized for low-end devices and poor network conditions
4. **Scalability**: Horizontal scaling to support growing user base
5. **Data Privacy**: Farmer data protection and compliance with regulations
6. **Continuous Learning**: ML models that improve with real-world feedback
7. **Cost Efficiency**: Optimize API usage and cloud resources to stay within budget

### Technology Stack

- **Frontend**: React.js with Next.js for SSR/SSG, PWA capabilities
- **Backend**: Node.js for API services, Python FastAPI for ML services
- **Database**: Supabase (PostgreSQL) with PostGIS for geospatial data
- **ML Framework**: Scikit-learn, XGBoost for tabular data; TensorFlow/PyTorch for deep learning
- **LLM Integration**: OpenAI GPT-4 for conversational AI and decision support
- **Message Queue**: Redis for caching and job queues
- **Container Orchestration**: Docker + Kubernetes for deployment
- **Cloud Platform**: AWS (primary) with multi-region support
- **Monitoring**: Prometheus + Grafana for metrics, ELK stack for logs
- **CDN**: CloudFront for static assets

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   PWA Web    │  │ Voice Client │  │ SMS Gateway  │          │
│  │   Interface  │  │  (Regional)  │  │   (Alerts)   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API Gateway Layer                           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  API Gateway (Rate Limiting, Auth, Routing, Logging)      │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Application Services                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │    Auth      │  │   Farmer     │  │  Extension   │          │
│  │   Service    │  │   Service    │  │   Officer    │          │
│  └──────────────┘  └──────────────┘  │   Service    │          │
│  ┌──────────────┐  ┌──────────────┐  └──────────────┘          │
│  │  Prediction  │  │Recommendation│  ┌──────────────┐          │
│  │   Service    │  │   Service    │  │    Alert     │          │
│  └──────────────┘  └──────────────┘  │   Service    │          │
│                                       └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Data & ML Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  ML Pipeline │  │   Satellite  │  │    Data      │          │
│  │   Service    │  │   Monitor    │  │  Collector   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    External Integrations                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Weather    │  │  Google Earth│  │  Government  │          │
│  │     APIs     │  │    Engine    │  │     APIs     │          │
│  │ (OpenWeather,│  │              │  │  (PM-KISAN,  │          │
│  │     IMD)     │  │              │  │  Soil Health)│          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │Agromonitoring│  │   OpenAI     │  │  SMS Gateway │          │
│  │     API      │  │   GPT-4 API  │  │   (Twilio)   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Data Storage Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  PostgreSQL  │  │     Redis    │  │   S3/Blob    │          │
│  │  (Supabase)  │  │    Cache     │  │   Storage    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### Service Responsibilities

**API Gateway**: Single entry point, authentication, rate limiting, request routing, logging
**Auth Service**: User registration, login, JWT management, OTP verification, RBAC
**Farmer Service**: Profile management, field registration, dashboard data aggregation
**Prediction Service**: Orchestrates data collection and ML inference for yield predictions
**Recommendation Service**: Generates personalized farming recommendations
**Alert Service**: Manages notifications via SMS, push, and voice calls
**ML Pipeline Service**: Model training, validation, deployment, and monitoring
**Satellite Monitor**: Fetches and processes satellite imagery from Google Earth Engine
**Data Collector**: Aggregates data from weather, soil, and external APIs
**Extension Officer Service**: Dashboard and tools for agricultural extension officers

## Components and Interfaces

### 1. API Gateway

**Responsibilities**: Request routing, authentication, rate limiting, logging, error handling

**Endpoints**:

```
POST   /api/v1/auth/register          - User registration
POST   /api/v1/auth/login             - User login
POST   /api/v1/auth/verify-otp        - OTP verification
POST   /api/v1/auth/refresh-token     - Token refresh
POST   /api/v1/auth/reset-password    - Password reset

GET    /api/v1/farmers/profile        - Get farmer profile
PUT    /api/v1/farmers/profile        - Update farmer profile
POST   /api/v1/farmers/fields         - Register new field
GET    /api/v1/farmers/fields         - List farmer's fields
GET    /api/v1/farmers/fields/:id     - Get field details

POST   /api/v1/predictions/yield      - Request yield prediction
GET    /api/v1/predictions/:id        - Get prediction details
GET    /api/v1/predictions/history    - Get prediction history

POST   /api/v1/recommendations        - Get farming recommendations
GET    /api/v1/recommendations/:id    - Get recommendation details

GET    /api/v1/satellite/imagery/:fieldId  - Get satellite imagery
GET    /api/v1/satellite/ndvi/:fieldId     - Get NDVI analysis

GET    /api/v1/weather/forecast       - Get weather forecast
GET    /api/v1/weather/current        - Get current weather

GET    /api/v1/soil/data/:location    - Get soil health data
POST   /api/v1/soil/manual-entry      - Submit manual soil test

GET    /api/v1/schemes                - List government schemes
GET    /api/v1/schemes/:id            - Get scheme details
GET    /api/v1/schemes/eligible       - Get eligible schemes

POST   /api/v1/voice/query            - Process voice query
POST   /api/v1/voice/transcribe       - Transcribe audio

GET    /api/v1/alerts                 - Get user alerts
PUT    /api/v1/alerts/:id/read        - Mark alert as read
PUT    /api/v1/alerts/preferences     - Update alert preferences

GET    /api/v1/extension/dashboard    - Extension officer dashboard
GET    /api/v1/extension/farmers      - List assigned farmers
POST   /api/v1/extension/notes        - Add farmer notes
POST   /api/v1/extension/broadcast    - Broadcast message

POST   /api/v1/sync/upload            - Upload offline data
GET    /api/v1/sync/download          - Download updates
```

**Rate Limiting**:
- Standard endpoints: 100 requests/minute/user
- ML inference endpoints: 20 requests/minute/user
- Voice endpoints: 30 requests/minute/user
- Public endpoints: 1000 requests/minute/IP

**Authentication**: JWT tokens with 30-day expiry, refresh token rotation

### 2. Authentication Service

**Interface**:
```typescript
interface AuthService {
  register(mobile: string, name: string, language: string): Promise<{userId: string, otpSent: boolean}>
  verifyOTP(userId: string, otp: string): Promise<{token: string, refreshToken: string}>
  login(mobile: string, password: string): Promise<{token: string, refreshToken: string}>
  refreshToken(refreshToken: string): Promise<{token: string}>
  resetPassword(mobile: string): Promise<{otpSent: boolean}>
  validateToken(token: string): Promise<{valid: boolean, userId: string, role: string}>
}
```

**Implementation Details**:
- OTP generation: 6-digit random number, valid for 5 minutes
- Password hashing: bcrypt with 12 rounds
- JWT payload: `{userId, role, mobile, iat, exp}`
- Token storage: Redis with TTL for blacklisting
- Account lockout: 5 failed attempts = 30-minute lock

### 3. Prediction Service

**Interface**:
```typescript
interface PredictionService {
  predictYield(request: YieldPredictionRequest): Promise<YieldPrediction>
  getPredictionHistory(farmerId: string, fieldId?: string): Promise<YieldPrediction[]>
  updateActualYield(predictionId: string, actualYield: number): Promise<void>
}

interface YieldPredictionRequest {
  farmerId: string
  fieldId: string
  cropType: string
  sowingDate: Date
  location: {latitude: number, longitude: number}
}

interface YieldPrediction {
  predictionId: string
  cropType: string
  predictedYield: number  // in quintals per acre
  confidenceInterval: {lower: number, upper: number}
  confidence: number  // 0-100
  factors: {
    weather: number
    soil: number
    satellite: number
    historical: number
  }
  harvestDate: Date
  createdAt: Date
}
```

**ML Model Pipeline**:
1. **Data Collection**: Aggregate weather (30 days historical + 90 days forecast), soil parameters, satellite imagery (NDVI, EVI, soil moisture), historical yields
2. **Feature Engineering**: Calculate growing degree days, rainfall accumulation, NDVI trends, soil nutrient scores
3. **Model Ensemble**: Combine Random Forest (70% weight), XGBoost (20%), LSTM (10%) predictions
4. **Confidence Calculation**: Based on data completeness, model agreement, historical accuracy
5. **Post-processing**: Convert to local units, apply regional calibration factors

### 4. Recommendation Engine

**Interface**:
```typescript
interface RecommendationService {
  generateRecommendations(request: RecommendationRequest): Promise<Recommendations>
  getRecommendationHistory(farmerId: string): Promise<Recommendations[]>
  provideFeedback(recommendationId: string, feedback: Feedback): Promise<void>
}

interface RecommendationRequest {
  farmerId: string
  fieldId: string
  cropType: string
  growthStage: string
  currentDate: Date
}

interface Recommendations {
  recommendationId: string
  irrigation: IrrigationSchedule
  fertilization: FertilizerPlan
  pestControl: PestManagement
  schemes: GovernmentScheme[]
  language: string
  createdAt: Date
}

interface IrrigationSchedule {
  nextIrrigation: Date
  amount: number  // in mm or liters
  frequency: string
  method: string
  reasoning: string
}

interface FertilizerPlan {
  applications: Array<{
    date: Date
    type: string  // NPK, Urea, etc.
    quantity: number  // in kg per acre
    method: string
  }>
  reasoning: string
}

interface PestManagement {
  risks: Array<{
    pest: string
    probability: number
    severity: string
    preventiveMeasures: string[]
  }>
  monitoring: string[]
}
```

**Recommendation Logic**:
1. **Irrigation**: Based on soil moisture (satellite), weather forecast, crop water requirements, growth stage
2. **Fertilization**: Based on soil health data, crop nutrient requirements, growth stage, yield target
3. **Pest Control**: Based on weather patterns, historical pest data, crop vulnerability, neighboring field reports
4. **Scheme Matching**: Based on farmer profile, crop type, land size, location, scheme eligibility criteria

### 5. Satellite Monitor Service

**Interface**:
```typescript
interface SatelliteMonitor {
  registerField(fieldId: string, boundary: GeoJSON): Promise<void>
  fetchLatestImagery(fieldId: string): Promise<SatelliteData>
  analyzeFieldHealth(fieldId: string): Promise<HealthAnalysis>
  getHistoricalTrends(fieldId: string, startDate: Date, endDate: Date): Promise<TrendData>
}

interface SatelliteData {
  fieldId: string
  acquisitionDate: Date
  cloudCover: number
  ndvi: {mean: number, min: number, max: number, stdDev: number}
  evi: {mean: number, min: number, max: number}
  soilMoisture: {mean: number, min: number, max: number}
  imageUrl: string
  ndviMapUrl: string
}

interface HealthAnalysis {
  overallHealth: 'excellent' | 'good' | 'moderate' | 'poor' | 'critical'
  stressAreas: Array<{
    location: {lat: number, lon: number}
    severity: number
    possibleCauses: string[]
  }>
  recommendations: string[]
  alertLevel: 'none' | 'low' | 'medium' | 'high' | 'critical'
}
```

**Google Earth Engine Integration**:
- **Satellite Sources**: Sentinel-2 (10m resolution, 5-day revisit), Landsat 8 (30m resolution, 16-day revisit)
- **Indices Calculated**: NDVI, EVI, NDWI (water), SAVI (soil-adjusted)
- **Processing**: Cloud masking, temporal compositing, zonal statistics
- **Thresholds**: NDVI < 0.3 (stress), NDVI 0.3-0.6 (moderate), NDVI > 0.6 (healthy)
- **Update Frequency**: Check for new imagery every 24 hours, process within 1 hour of availability

### 6. Voice Assistant Service

**Interface**:
```typescript
interface VoiceAssistant {
  transcribe(audioData: Buffer, language: string): Promise<{text: string, confidence: number}>
  processQuery(text: string, language: string, context: UserContext): Promise<Response>
  synthesize(text: string, language: string): Promise<Buffer>
}

interface UserContext {
  farmerId: string
  currentFields: string[]
  recentQueries: string[]
  preferences: {language: string, voiceGender: string}
}

interface Response {
  text: string
  audio: Buffer
  actions: Array<{type: string, data: any}>
  followUpQuestions: string[]
}
```

**NLP Pipeline**:
1. **Speech-to-Text**: Use regional language ASR models (Google Speech API, Azure Speech, or custom models)
2. **Intent Classification**: Classify query into categories (yield_prediction, weather, recommendation, scheme_info, pest_identification)
3. **Entity Extraction**: Extract crop names, dates, locations, quantities
4. **Query Processing**: Route to appropriate service based on intent
5. **LLM Enhancement**: For complex queries, use OpenAI GPT-4 for intelligent response generation
6. **Response Generation**: Combine structured data with LLM-generated explanations
7. **Text-to-Speech**: Convert response to speech in regional language

**OpenAI Integration Details**:

```typescript
interface OpenAIService {
  interpretQuery(text: string, language: string, context: UserContext): Promise<QueryIntent>
  generateResponse(query: string, data: any, language: string): Promise<string>
  explainRecommendation(recommendation: Recommendation, language: string): Promise<string>
  searchKnowledgeBase(query: string): Promise<KnowledgeResult[]>
}
```

**OpenAI API Usage**:
- **Chat Completions**: For conversational AI and complex question answering
- **Embeddings**: For semantic search in agricultural knowledge base
- **Fine-tuning**: Custom models trained on agricultural domain data

**Use Cases**:
1. **Voice Query Interpretation**: Extract intent and entities from natural language
2. **Complex Question Answering**: "Why is my wheat crop showing yellow leaves in January?"
3. **Recommendation Explanation**: Convert technical recommendations to farmer-friendly language
4. **Knowledge Augmentation**: Search agricultural research papers and best practices
5. **Multi-turn Conversations**: Maintain context for follow-up questions

**Cost Optimization**:
- Cache common query responses (Redis, 24-hour TTL)
- Use embeddings for knowledge base search (one-time cost)
- Implement query classification to route simple queries to rule-based system
- Set token limits per query (max 500 tokens for responses)
- Monitor API usage and implement daily/monthly caps

**Example OpenAI Integration**:

```typescript
async function processComplexQuery(query: string, farmerContext: FarmerContext): Promise<Response> {
  // Check cache first
  const cached = await redis.get(`query:${hash(query)}`);
  if (cached) return JSON.parse(cached);
  
  // Get farmer's field data for context
  const fieldData = await getFieldData(farmerContext.fieldId);
  const weatherData = await getWeatherData(farmerContext.location);
  
  // Build context for OpenAI
  const systemPrompt = `You are an agricultural expert helping Indian farmers. 
    Farmer's context: Crop=${fieldData.cropType}, Location=${farmerContext.location}, 
    Current weather=${weatherData.summary}. 
    Provide practical advice in simple language.`;
  
  // Call OpenAI
  const response = await openai.chat.completions.create({
    model: "gpt-4",
    messages: [
      {role: "system", content: systemPrompt},
      {role: "user", content: query}
    ],
    max_tokens: 500,
    temperature: 0.7
  });
  
  const answer = response.choices[0].message.content;
  
  // Translate to regional language if needed
  const translated = await translateToRegionalLanguage(answer, farmerContext.language);
  
  // Cache for 24 hours
  await redis.setex(`query:${hash(query)}`, 86400, JSON.stringify(translated));
  
  return translated;
}
```

**Supported Intents**:
- Yield prediction queries
- Weather information
- Irrigation advice
- Fertilizer recommendations
- Pest identification and control
- Government scheme information
- Market prices
- General farming advice
- Crop disease diagnosis
- Soil health interpretation

### 7. Data Collector Service

**Interface**:
```typescript
interface DataCollector {
  collectWeatherData(location: Location, startDate: Date, endDate: Date): Promise<WeatherData[]>
  collectSoilData(location: Location): Promise<SoilData>
  collectHistoricalYields(location: Location, cropType: string): Promise<HistoricalYield[]>
  aggregateDataForPrediction(request: DataRequest): Promise<AggregatedData>
}

interface WeatherData {
  date: Date
  temperature: {min: number, max: number, avg: number}
  rainfall: number
  humidity: number
  windSpeed: number
  solarRadiation: number
  source: string
}

interface SoilData {
  location: Location
  pH: number
  nitrogen: number  // kg/ha
  phosphorus: number
  potassium: number
  organicCarbon: number
  texture: string
  lastUpdated: Date
  source: string
}
```

**Data Sources**:
- **Weather**: OpenWeatherMap API (primary), India Meteorological Department (IMD) API (secondary), Agromonitoring API (tertiary)
- **Soil**: Soil Health Card portal, state agriculture department APIs, manual farmer input
- **Satellite**: Google Earth Engine (Sentinel-2, Landsat 8), Agromonitoring API (backup)
- **Historical Yields**: Government agricultural statistics, farmer-reported yields

**Agromonitoring API Integration**:

```typescript
interface AgromonitoringService {
  getFieldPolygon(fieldId: string): Promise<PolygonData>
  getSatelliteImagery(polygonId: string, startDate: Date, endDate: Date): Promise<ImageryData[]>
  getWeatherForecast(polygonId: string): Promise<WeatherForecast>
  getNDVI(polygonId: string, date: Date): Promise<NDVIData>
  getSoilMoisture(polygonId: string, date: Date): Promise<SoilMoistureData>
}
```

**Agromonitoring Features**:
- **Satellite Imagery**: Sentinel-2 data with NDVI, EVI, NDWI indices
- **Weather Data**: 16-day forecasts with temperature, precipitation, wind
- **Soil Data**: Soil moisture and temperature from satellite
- **Historical Data**: Archive of satellite imagery for trend analysis

**Data Source Prioritization**:
1. **Primary**: Google Earth Engine (higher resolution, more processing options)
2. **Backup**: Agromonitoring (when GEE unavailable or rate limited)
3. **Fallback**: Cached data with staleness indicator

**Multi-Source Aggregation**:
```typescript
async function aggregateWeatherData(location: Location): Promise<WeatherData> {
  const sources = await Promise.allSettled([
    openWeatherAPI.getForecast(location),
    imdAPI.getForecast(location),
    agromonitoringAPI.getForecast(location)
  ]);
  
  const validSources = sources
    .filter(s => s.status === 'fulfilled')
    .map(s => s.value);
  
  if (validSources.length === 0) {
    return getCachedWeatherData(location);
  }
  
  // Weighted average based on source reliability
  return {
    temperature: weightedAverage(validSources.map(s => s.temperature), [0.5, 0.3, 0.2]),
    rainfall: weightedAverage(validSources.map(s => s.rainfall), [0.5, 0.3, 0.2]),
    humidity: weightedAverage(validSources.map(s => s.humidity), [0.5, 0.3, 0.2]),
    sources: validSources.map(s => s.source)
  };
}
```

**Caching Strategy**:
- Weather forecasts: 6-hour TTL
- Historical weather: 24-hour TTL
- Soil data: 30-day TTL (updated annually)
- Satellite imagery: 7-day TTL
- Historical yields: 90-day TTL
- Agromonitoring data: Same as primary sources

### 8. Alert Service

**Interface**:
```typescript
interface AlertService {
  createAlert(alert: Alert): Promise<string>
  sendAlert(alertId: string, channels: Channel[]): Promise<DeliveryStatus>
  getAlerts(farmerId: string, filters: AlertFilters): Promise<Alert[]>
  markAsRead(alertId: string): Promise<void>
  updatePreferences(farmerId: string, preferences: AlertPreferences): Promise<void>
}

interface Alert {
  alertId: string
  farmerId: string
  fieldId?: string
  type: 'weather' | 'pest' | 'irrigation' | 'harvest' | 'scheme' | 'system'
  severity: 'low' | 'medium' | 'high' | 'critical'
  title: string
  message: string
  language: string
  actionable: boolean
  actions?: Array<{label: string, action: string}>
  expiresAt?: Date
  createdAt: Date
}

interface AlertPreferences {
  channels: {
    sms: boolean
    push: boolean
    voice: boolean
  }
  types: {
    weather: boolean
    pest: boolean
    irrigation: boolean
    harvest: boolean
    scheme: boolean
  }
  quietHours: {start: string, end: string}
}
```

**Delivery Channels**:
- **SMS**: Twilio/AWS SNS, 160 characters, regional language support
- **Push Notifications**: Firebase Cloud Messaging, rich notifications with images
- **Voice Calls**: Twilio Voice, automated calls for critical alerts
- **In-App**: Real-time via WebSocket, persistent storage

**Alert Triggers**:
- Weather: Extreme events forecasted (heavy rain, heatwave, frost)
- Pest: High risk detected based on weather and historical data
- Irrigation: Soil moisture below threshold
- Harvest: Optimal harvest window approaching
- Scheme: New eligible scheme or application deadline
- System: Service disruptions, maintenance windows

### 9. ML Pipeline Service

**Interface**:
```typescript
interface MLPipeline {
  trainModel(config: TrainingConfig): Promise<ModelMetrics>
  validateModel(modelId: string, validationData: Dataset): Promise<ValidationResults>
  deployModel(modelId: string, environment: 'staging' | 'production'): Promise<DeploymentStatus>
  rollbackModel(modelId: string): Promise<void>
  getModelMetrics(modelId: string): Promise<ModelMetrics>
}

interface TrainingConfig {
  modelType: 'random_forest' | 'xgboost' | 'lstm' | 'ensemble'
  cropTypes: string[]
  regions: string[]
  features: string[]
  hyperparameters: Record<string, any>
  trainingData: {
    startDate: Date
    endDate: Date
    minSamples: number
  }
}

interface ModelMetrics {
  modelId: string
  accuracy: number
  rmse: number
  mae: number
  r2Score: number
  cropTypeMetrics: Record<string, {accuracy: number, rmse: number}>
  featureImportance: Record<string, number>
  trainedAt: Date
}
```

**ML Architecture**:

**Feature Engineering**:
```python
features = {
    # Weather features
    'gdd': growing_degree_days,  # Accumulated temperature above base
    'rainfall_total': sum(rainfall),
    'rainfall_distribution': coefficient_of_variation(rainfall),
    'extreme_temp_days': count(temp > threshold),
    
    # Soil features
    'soil_fertility_index': weighted_avg(N, P, K, OC),
    'soil_texture_encoded': one_hot_encode(texture),
    'ph_optimal_distance': abs(pH - crop_optimal_pH),
    
    # Satellite features
    'ndvi_mean': mean(ndvi_timeseries),
    'ndvi_trend': linear_regression_slope(ndvi_timeseries),
    'ndvi_variability': std(ndvi_timeseries),
    'evi_peak': max(evi_timeseries),
    
    # Historical features
    'avg_yield_3yr': mean(yields[-3:]),
    'yield_trend': linear_regression_slope(yields),
    'yield_variability': std(yields),
    
    # Temporal features
    'sowing_month': month(sowing_date),
    'season': encode_season(sowing_date),
    'days_to_harvest': crop_duration
}
```

**Model Ensemble**:
1. **Random Forest**: 500 trees, max_depth=20, handles non-linear relationships
2. **XGBoost**: learning_rate=0.1, n_estimators=300, captures complex interactions
3. **LSTM**: 2 layers, 128 units, for temporal patterns in weather/NDVI
4. **Ensemble**: Weighted average (RF: 0.7, XGB: 0.2, LSTM: 0.1) based on validation performance

**Training Pipeline**:
1. Data collection from database (weather, soil, satellite, yields)
2. Data cleaning and outlier removal
3. Feature engineering
4. Train-validation-test split (70-15-15)
5. Model training with cross-validation
6. Hyperparameter tuning using Optuna
7. Model validation on holdout set
8. Model serialization and versioning
9. Deployment to staging for A/B testing
10. Production deployment after validation

**Retraining Schedule**: Monthly with new data, triggered automatically

### 10. Sync Manager (Offline Support)

**Interface**:
```typescript
interface SyncManager {
  cacheEssentialData(farmerId: string): Promise<CachedData>
  recordOfflineAction(action: OfflineAction): Promise<string>
  syncToCloud(): Promise<SyncResult>
  resolveConflicts(conflicts: Conflict[]): Promise<Resolution[]>
  getPendingActions(): Promise<OfflineAction[]>
}

interface OfflineAction {
  actionId: string
  type: 'create' | 'update' | 'delete'
  entity: string
  data: any
  timestamp: Date
  synced: boolean
}

interface SyncResult {
  success: boolean
  syncedActions: number
  failedActions: number
  conflicts: Conflict[]
}
```

**Offline Strategy**:
- **Service Worker**: Cache API responses, static assets, and app shell
- **IndexedDB**: Store user data, predictions, recommendations, alerts
- **Background Sync**: Automatically sync when connection restored
- **Conflict Resolution**: Last-write-wins with user notification for critical data

**Cached Data**:
- Last 7 days of predictions and recommendations
- Current field information and crop status
- Last 30 days of alerts
- Crop information database
- Regional language translations
- User preferences and settings

## Data Models

### Database Schema

**Users Table**:
```sql
CREATE TABLE users (
  user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  mobile VARCHAR(15) UNIQUE NOT NULL,
  name VARCHAR(100) NOT NULL,
  role VARCHAR(20) NOT NULL CHECK (role IN ('farmer', 'extension_officer', 'admin')),
  language VARCHAR(10) NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  otp VARCHAR(6),
  otp_expires_at TIMESTAMP,
  failed_login_attempts INT DEFAULT 0,
  locked_until TIMESTAMP,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW(),
  last_login TIMESTAMP
);

CREATE INDEX idx_users_mobile ON users(mobile);
CREATE INDEX idx_users_role ON users(role);
```

**Farmers Table**:
```sql
CREATE TABLE farmers (
  farmer_id UUID PRIMARY KEY REFERENCES users(user_id),
  village VARCHAR(100),
  district VARCHAR(100),
  state VARCHAR(100),
  land_size_acres DECIMAL(10, 2),
  farming_experience_years INT,
  education_level VARCHAR(50),
  extension_officer_id UUID REFERENCES users(user_id),
  preferences JSONB,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_farmers_district ON farmers(district);
CREATE INDEX idx_farmers_extension_officer ON farmers(extension_officer_id);
```

**Fields Table**:
```sql
CREATE TABLE fields (
  field_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  farmer_id UUID NOT NULL REFERENCES farmers(farmer_id),
  name VARCHAR(100),
  area_acres DECIMAL(10, 2) NOT NULL,
  location GEOGRAPHY(POINT, 4326) NOT NULL,
  boundary GEOGRAPHY(POLYGON, 4326),
  soil_type VARCHAR(50),
  irrigation_type VARCHAR(50),
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_fields_farmer ON fields(farmer_id);
CREATE INDEX idx_fields_location ON fields USING GIST(location);
```

**Crops Table**:
```sql
CREATE TABLE crops (
  crop_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  field_id UUID NOT NULL REFERENCES fields(field_id),
  crop_type VARCHAR(50) NOT NULL,
  variety VARCHAR(100),
  sowing_date DATE NOT NULL,
  expected_harvest_date DATE,
  actual_harvest_date DATE,
  growth_stage VARCHAR(50),
  status VARCHAR(20) CHECK (status IN ('active', 'harvested', 'failed')),
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_crops_field ON crops(field_id);
CREATE INDEX idx_crops_type ON crops(crop_type);
CREATE INDEX idx_crops_status ON crops(status);
```

**Predictions Table**:
```sql
CREATE TABLE predictions (
  prediction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  crop_id UUID NOT NULL REFERENCES crops(crop_id),
  farmer_id UUID NOT NULL REFERENCES farmers(farmer_id),
  predicted_yield DECIMAL(10, 2) NOT NULL,
  confidence_lower DECIMAL(10, 2),
  confidence_upper DECIMAL(10, 2),
  confidence_score DECIMAL(5, 2),
  actual_yield DECIMAL(10, 2),
  model_version VARCHAR(50) NOT NULL,
  features JSONB,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_predictions_crop ON predictions(crop_id);
CREATE INDEX idx_predictions_farmer ON predictions(farmer_id);
CREATE INDEX idx_predictions_created ON predictions(created_at DESC);
```

**Recommendations Table**:
```sql
CREATE TABLE recommendations (
  recommendation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  crop_id UUID NOT NULL REFERENCES crops(crop_id),
  farmer_id UUID NOT NULL REFERENCES farmers(farmer_id),
  type VARCHAR(50) NOT NULL,
  content JSONB NOT NULL,
  language VARCHAR(10) NOT NULL,
  status VARCHAR(20) DEFAULT 'pending',
  feedback JSONB,
  created_at TIMESTAMP DEFAULT NOW(),
  expires_at TIMESTAMP
);

CREATE INDEX idx_recommendations_crop ON recommendations(crop_id);
CREATE INDEX idx_recommendations_farmer ON recommendations(farmer_id);
CREATE INDEX idx_recommendations_status ON recommendations(status);
```

**Satellite Data Table**:
```sql
CREATE TABLE satellite_data (
  satellite_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  field_id UUID NOT NULL REFERENCES fields(field_id),
  acquisition_date DATE NOT NULL,
  satellite_source VARCHAR(50),
  cloud_cover DECIMAL(5, 2),
  ndvi_mean DECIMAL(5, 3),
  ndvi_min DECIMAL(5, 3),
  ndvi_max DECIMAL(5, 3),
  ndvi_stddev DECIMAL(5, 3),
  evi_mean DECIMAL(5, 3),
  soil_moisture DECIMAL(5, 2),
  image_url TEXT,
  ndvi_map_url TEXT,
  analysis JSONB,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_satellite_field ON satellite_data(field_id);
CREATE INDEX idx_satellite_date ON satellite_data(acquisition_date DESC);
CREATE UNIQUE INDEX idx_satellite_field_date ON satellite_data(field_id, acquisition_date);
```

**Weather Data Table**:
```sql
CREATE TABLE weather_data (
  weather_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  location GEOGRAPHY(POINT, 4326) NOT NULL,
  date DATE NOT NULL,
  temp_min DECIMAL(5, 2),
  temp_max DECIMAL(5, 2),
  temp_avg DECIMAL(5, 2),
  rainfall DECIMAL(6, 2),
  humidity DECIMAL(5, 2),
  wind_speed DECIMAL(5, 2),
  solar_radiation DECIMAL(6, 2),
  source VARCHAR(50),
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_weather_location ON weather_data USING GIST(location);
CREATE INDEX idx_weather_date ON weather_data(date DESC);
CREATE UNIQUE INDEX idx_weather_location_date ON weather_data(ST_SnapToGrid(location::geometry, 0.01), date);
```

**Soil Data Table**:
```sql
CREATE TABLE soil_data (
  soil_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  location GEOGRAPHY(POINT, 4326) NOT NULL,
  ph DECIMAL(4, 2),
  nitrogen DECIMAL(8, 2),
  phosphorus DECIMAL(8, 2),
  potassium DECIMAL(8, 2),
  organic_carbon DECIMAL(5, 2),
  texture VARCHAR(50),
  source VARCHAR(100),
  test_date DATE,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_soil_location ON soil_data USING GIST(location);
CREATE INDEX idx_soil_test_date ON soil_data(test_date DESC);
```

**Alerts Table**:
```sql
CREATE TABLE alerts (
  alert_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  farmer_id UUID NOT NULL REFERENCES farmers(farmer_id),
  field_id UUID REFERENCES fields(field_id),
  type VARCHAR(50) NOT NULL,
  severity VARCHAR(20) NOT NULL,
  title VARCHAR(200) NOT NULL,
  message TEXT NOT NULL,
  language VARCHAR(10) NOT NULL,
  actionable BOOLEAN DEFAULT FALSE,
  actions JSONB,
  read BOOLEAN DEFAULT FALSE,
  delivered_channels JSONB,
  expires_at TIMESTAMP,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_alerts_farmer ON alerts(farmer_id);
CREATE INDEX idx_alerts_created ON alerts(created_at DESC);
CREATE INDEX idx_alerts_read ON alerts(read) WHERE read = FALSE;
```

**Government Schemes Table**:
```sql
CREATE TABLE government_schemes (
  scheme_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name VARCHAR(200) NOT NULL,
  description TEXT,
  benefits TEXT,
  eligibility_criteria JSONB,
  application_process TEXT,
  application_url TEXT,
  start_date DATE,
  end_date DATE,
  scheme_type VARCHAR(50),
  level VARCHAR(20) CHECK (level IN ('central', 'state', 'district')),
  state VARCHAR(100),
  active BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_schemes_active ON government_schemes(active) WHERE active = TRUE;
CREATE INDEX idx_schemes_state ON government_schemes(state);
CREATE INDEX idx_schemes_type ON government_schemes(scheme_type);
```

**ML Models Table**:
```sql
CREATE TABLE ml_models (
  model_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  model_type VARCHAR(50) NOT NULL,
  version VARCHAR(50) NOT NULL,
  crop_types TEXT[],
  regions TEXT[],
  accuracy DECIMAL(5, 4),
  rmse DECIMAL(10, 4),
  mae DECIMAL(10, 4),
  r2_score DECIMAL(5, 4),
  metrics JSONB,
  hyperparameters JSONB,
  feature_importance JSONB,
  model_path TEXT NOT NULL,
  status VARCHAR(20) CHECK (status IN ('training', 'validation', 'staging', 'production', 'retired')),
  trained_at TIMESTAMP,
  deployed_at TIMESTAMP,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_models_status ON ml_models(status);
CREATE INDEX idx_models_version ON ml_models(version);
```

**Audit Logs Table**:
```sql
CREATE TABLE audit_logs (
  log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(user_id),
  action VARCHAR(100) NOT NULL,
  entity_type VARCHAR(50),
  entity_id UUID,
  changes JSONB,
  ip_address INET,
  user_agent TEXT,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_audit_user ON audit_logs(user_id);
CREATE INDEX idx_audit_created ON audit_logs(created_at DESC);
CREATE INDEX idx_audit_entity ON audit_logs(entity_type, entity_id);
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Authentication and Authorization Properties

**Property 1: User Registration Creates Account with OTP**
*For any* valid registration request with mobile number and user details, the Authentication Service should create a user account and generate an OTP that is sent to the provided mobile number.
**Validates: Requirements 1.1**

**Property 2: Valid OTP Verification Generates JWT**
*For any* valid OTP entered within the 5-minute expiration window, the Authentication Service should verify the account and return a valid JWT token with 30-day expiration.
**Validates: Requirements 1.2, 1.3**

**Property 3: Expired Tokens Are Rejected**
*For any* API request with an expired JWT token, the API Gateway should reject the request with 401 status and prompt for re-authentication.
**Validates: Requirements 1.4**

**Property 4: Role-Based Access Control Enforcement**
*For any* authenticated user with a specific role, the system should only allow access to resources and operations permitted for that role.
**Validates: Requirements 1.5**

**Property 5: Account Lockout After Failed Attempts**
*For any* user account with more than 5 failed login attempts within 15 minutes, the Authentication Service should lock the account for 30 minutes.
**Validates: Requirements 1.6**

**Property 6: Password Encryption with Bcrypt**
*For any* password stored in the database, it should be encrypted using bcrypt with minimum 12 rounds, and the original password should not be recoverable.
**Validates: Requirements 1.8, 13.3**

### Yield Prediction Properties

**Property 7: Data Collection for Predictions**
*For any* crop type and location coordinates provided by a farmer, the Data Collector should retrieve weather data, soil parameters, and satellite imagery for that location.
**Validates: Requirements 2.1**

**Property 8: Prediction Generation from Complete Data**
*For any* complete dataset (weather, soil, satellite), the Yield Predictor should generate a yield prediction with confidence interval within 5 seconds.
**Validates: Requirements 2.2**

**Property 9: Multi-Crop Support**
*For any* crop type from the supported list of 100+ crops, the Yield Predictor should accept the crop type and generate predictions.
**Validates: Requirements 2.6**

**Property 10: Low Confidence Warning Display**
*For any* yield prediction with confidence score below 70%, the system should display a warning message to the farmer.
**Validates: Requirements 2.7**

**Property 11: Local Unit Display**
*For any* yield prediction displayed to a farmer, the yield value should be shown in local units (quintals per acre or tonnes per hectare) based on farmer's region.
**Validates: Requirements 2.8**

### Recommendation Engine Properties

**Property 12: Irrigation Schedule Generation**
*For any* valid crop and location data with current soil moisture and weather forecast, the Recommendation Engine should generate an irrigation schedule.
**Validates: Requirements 3.1**

**Property 13: Fertilizer Recommendations for Deficiencies**
*For any* soil health data indicating nutrient deficiency (N, P, K below optimal ranges), the Recommendation Engine should suggest specific fertilizer types and quantities.
**Validates: Requirements 3.2**

**Property 14: Pest Risk Identification**
*For any* weather conditions that favor pest outbreaks (based on pest risk models), the Recommendation Engine should identify potential pest risks and recommend preventive measures.
**Validates: Requirements 3.3**

**Property 15: Regional Language Recommendations**
*For any* farmer with a selected regional language preference, all recommendations should be provided in that language.
**Validates: Requirements 3.6, 9.7**

**Property 16: Government Scheme Inclusion**
*For any* farmer whose profile matches government scheme eligibility criteria, the Recommendation Engine should include information about applicable schemes.
**Validates: Requirements 3.7, 17.1**

### Satellite Monitoring Properties

**Property 17: Automatic Satellite Monitoring Configuration**
*For any* field registered by a farmer with location coordinates, the Satellite Monitor should configure automatic satellite imagery retrieval from Google Earth Engine.
**Validates: Requirements 4.1**

**Property 18: NDVI Calculation from Imagery**
*For any* new satellite imagery available for a field, the Satellite Monitor should calculate NDVI values and soil moisture indices.
**Validates: Requirements 4.3**

**Property 19: Crop Stress Detection and Alerting**
*For any* field where NDVI values drop below crop-specific thresholds, the Satellite Monitor should detect potential crop stress and trigger an alert.
**Validates: Requirements 4.4**

**Property 20: Satellite Data Historical Storage**
*For any* processed satellite imagery and analysis, the system should store it in the historical record for trend analysis.
**Validates: Requirements 4.8**

### Voice Interface Properties

**Property 21: Multi-Language Voice Support**
*For any* of the 15+ supported regional languages, the Voice Assistant should be able to transcribe speech, process queries, and synthesize responses in that language.
**Validates: Requirements 5.1, 5.2, 5.5**

**Property 22: Voice Query Routing**
*For any* transcribed voice query, the Voice Assistant should classify the intent and route to the appropriate service (prediction, recommendation, weather, etc.).
**Validates: Requirements 5.4**

**Property 23: Voice Recognition Failure Handling**
*For any* voice input where recognition fails or confidence is low, the Voice Assistant should prompt the farmer to repeat the query.
**Validates: Requirements 5.7**

### Offline Functionality Properties

**Property 24: Essential Data Caching**
*For any* PWA installation, the system should cache essential data including recent predictions, recommendations, crop information, and alerts for offline access.
**Validates: Requirements 6.1, 19.4**

**Property 25: Offline Data Recording**
*For any* data recorded by a farmer while offline, the Sync Manager should store it locally with timestamps.
**Validates: Requirements 6.3**

**Property 26: Sync Conflict Resolution**
*For any* sync conflict detected when synchronizing offline data, the Sync Manager should resolve it using last-write-wins strategy and notify the user.
**Validates: Requirements 6.5**

**Property 27: Offline Status Indication**
*For any* time when the system is operating offline, a clear offline status indicator should be displayed to the user.
**Validates: Requirements 6.7**

**Property 28: Critical Data Sync Priority**
*For any* sync operation with both critical (predictions, alerts) and non-critical (analytics) data pending, critical data should be synchronized first.
**Validates: Requirements 6.8**

### Weather Integration Properties

**Property 29: Weather Data Retrieval**
*For any* farmer location registered in the system, the Data Collector should retrieve weather forecasts from integrated APIs.
**Validates: Requirements 7.1**

**Property 30: Seven-Day Forecast Display**
*For any* weather data retrieved, the system should display a 7-day forecast including temperature, rainfall, humidity, and wind speed.
**Validates: Requirements 7.3**

**Property 31: Extreme Weather Alerting**
*For any* weather forecast indicating extreme events (heavy rain, heatwave, frost), the Alert System should send advance warnings to affected farmers.
**Validates: Requirements 7.4**

**Property 32: Weather API Retry Logic**
*For any* failed weather API call, the system should retry with exponential backoff up to 3 attempts before falling back to cached data.
**Validates: Requirements 7.6, 7.7**

### Soil Data Properties

**Property 33: Soil Parameter Display**
*For any* soil health data retrieved or entered, the system should display all parameters including pH, nitrogen, phosphorus, potassium, organic carbon, and micronutrients.
**Validates: Requirements 8.2**

**Property 34: Manual Soil Data Entry Fallback**
*For any* field location where government soil data is unavailable, the system should allow farmers to manually input soil test results.
**Validates: Requirements 8.3**

**Property 35: Soil Amendment Recommendations**
*For any* soil parameters outside optimal ranges for the selected crop, the Recommendation Engine should suggest specific soil amendments.
**Validates: Requirements 8.4**

**Property 36: Soil Data Historical Tracking**
*For any* soil health data entry (retrieved or manual), the system should store it in historical records to track changes over time.
**Validates: Requirements 8.5**

### Dashboard and Visualization Properties

**Property 37: Dashboard Essential Elements**
*For any* farmer login, the dashboard should display current yield predictions, upcoming tasks, and recent alerts.
**Validates: Requirements 9.1**

**Property 38: Priority-Based Information Organization**
*For any* dashboard display, information should be organized by priority with critical alerts and time-sensitive recommendations at the top.
**Validates: Requirements 9.4**

**Property 39: Field-Specific Information Display**
*For any* field selected by a farmer, the system should display field-specific information including crop status, satellite imagery, and historical data.
**Validates: Requirements 9.5**

**Property 40: Multiple Field Support**
*For any* farmer account, the system should support registration and management of multiple fields with easy switching between them.
**Validates: Requirements 9.6**

### Alert System Properties

**Property 41: Critical Event Notification Generation**
*For any* critical event detected (pest outbreak risk, extreme weather, irrigation needed), the Alert System should generate a notification.
**Validates: Requirements 10.1**

**Property 42: Multi-Channel Alert Delivery**
*For any* alert generated, the Alert System should send notifications through configured channels (in-app, SMS, voice for critical alerts).
**Validates: Requirements 10.2**

**Property 43: SMS Character Limit and Language**
*For any* SMS alert sent, the message should be within 160 characters and in the farmer's selected regional language.
**Validates: Requirements 10.3**

**Property 44: Field-Specific Alert Identification**
*For any* farmer with multiple fields, alerts should clearly identify which specific field the alert pertains to.
**Validates: Requirements 10.5**

**Property 45: Alert Queueing and Retry**
*For any* alert delivery failure due to poor connectivity, the Alert System should queue the notification and retry delivery.
**Validates: Requirements 10.7**

**Property 46: Notification History Maintenance**
*For any* notification sent, it should be stored in the notification history accessible from the dashboard.
**Validates: Requirements 10.8**

### ML Pipeline Properties

**Property 47: Training Data Incorporation**
*For any* new farming outcome data collected, the ML Pipeline should incorporate it into the training dataset for future model retraining.
**Validates: Requirements 11.1**

**Property 48: Model Validation Before Deployment**
*For any* retrained model, the ML Pipeline should validate it against holdout datasets and only deploy if accuracy meets the 85% threshold.
**Validates: Requirements 11.3**

**Property 49: Model Architecture Support**
*For any* of the supported model architectures (Random Forest, XGBoost, LSTM), the ML Pipeline should be able to train and deploy models of that type.
**Validates: Requirements 11.4**

**Property 50: Model Performance Degradation Alerting**
*For any* deployed model where performance drops below 80% accuracy, the ML Pipeline should trigger alerts to administrators.
**Validates: Requirements 11.7**

**Property 51: Model Versioning and Tracking**
*For any* model training run, the ML Pipeline should maintain version information and experiment tracking for reproducibility.
**Validates: Requirements 11.8**

### API Gateway Properties

**Property 52: Rate Limiting Enforcement**
*For any* user making API requests, the API Gateway should enforce rate limiting of 100 requests per minute for standard endpoints.
**Validates: Requirements 12.2**

**Property 53: Rate Limit Exceeded Response**
*For any* user exceeding rate limits, the API Gateway should return HTTP 429 status with retry-after header.
**Validates: Requirements 12.3**

**Property 54: JWT Validation for Authenticated Endpoints**
*For any* request to an authenticated endpoint, the API Gateway should validate the JWT token before routing to backend services.
**Validates: Requirements 12.4**

**Property 55: API Request Logging**
*For any* API request received, the API Gateway should log it with timestamp, user ID, endpoint, and response code.
**Validates: Requirements 12.5**

**Property 56: Standardized Error Responses**
*For any* failed API request, the API Gateway should return a standardized error response with error code and message.
**Validates: Requirements 12.6**

**Property 57: Request Timeout Enforcement**
*For any* API request exceeding 30 seconds, the API Gateway should timeout the request and return an appropriate error.
**Validates: Requirements 12.8**

### Security and Privacy Properties

**Property 58: TLS Encryption for Data in Transit**
*For any* data transmitted between client and server, the system should use TLS 1.3 or higher encryption.
**Validates: Requirements 13.1**

**Property 59: Sensitive Data Encryption at Rest**
*For any* sensitive data stored (personal information, location data, credentials), it should be encrypted at rest.
**Validates: Requirements 13.2**

**Property 60: RBAC Data Access Restriction**
*For any* data access request, the system should enforce role-based access control to ensure users can only access data permitted for their role.
**Validates: Requirements 13.4**

**Property 61: Account Deletion Data Removal**
*For any* farmer account deletion request, the system should permanently remove all personal data within 30 days.
**Validates: Requirements 13.5**

**Property 62: Third-Party Data Sharing Consent**
*For any* attempt to share farmer data with third parties, the system should verify explicit consent was granted before sharing.
**Validates: Requirements 13.7**

**Property 63: Audit Log Creation**
*For any* data access or modification operation, the system should create an audit log entry with user, action, timestamp, and changes.
**Validates: Requirements 13.8**

### Monitoring and Observability Properties

**Property 64: Metrics Collection**
*For any* system operation, the monitoring system should collect metrics including API response times, error rates, CPU usage, memory usage, and database query performance.
**Validates: Requirements 15.1**

**Property 65: Threshold-Based Administrator Alerts**
*For any* monitoring metric exceeding defined thresholds (error rate > 5%, response time > 5s), the system should send alerts to administrators.
**Validates: Requirements 15.2**

**Property 66: Structured Logging Format**
*For any* log entry created by any service, it should be in structured JSON format.
**Validates: Requirements 15.3**

**Property 67: Distributed Tracing**
*For any* request that spans multiple microservices, the system should create a distributed trace to track the request flow.
**Validates: Requirements 15.4**

**Property 68: Log Retention**
*For any* log entry created, it should be retained for minimum 30 days.
**Validates: Requirements 15.6**

**Property 69: Third-Party API Monitoring**
*For any* third-party API integrated with the system, availability and response times should be monitored.
**Validates: Requirements 15.7**

**Property 70: Automated Recovery Triggering**
*For any* critical service failure detected, the system should trigger automated recovery procedures and notify administrators.
**Validates: Requirements 15.8**

### Backup and Disaster Recovery Properties

**Property 71: Automated Backup Scheduling**
*For any* 6-hour period, the system should perform an automated database backup.
**Validates: Requirements 16.1**

**Property 72: Backup Retention Policy**
*For any* backup created, it should be retained according to policy: daily backups for 30 days, monthly backups for 1 year.
**Validates: Requirements 16.2**

**Property 73: Geographic Backup Separation**
*For any* backup stored, it should be in a geographically separate region from the primary database.
**Validates: Requirements 16.3**

**Property 74: Database Failover**
*For any* primary database failure detected, the system should automatically failover to replica databases.
**Validates: Requirements 16.5**

**Property 75: Replication Lag Monitoring**
*For any* database replication operation, the lag should be monitored and maintained at maximum 5 seconds.
**Validates: Requirements 16.6**

**Property 76: Backup Encryption**
*For any* backup data stored, it should be encrypted using AES-256 encryption.
**Validates: Requirements 16.8**

### Government Integration Properties

**Property 77: Scheme Information Display**
*For any* government scheme displayed to a farmer, it should show scheme name, benefits, eligibility criteria, and application process in the farmer's regional language.
**Validates: Requirements 17.3**

**Property 78: Scheme Database Maintenance**
*For any* query for government schemes, the system should have an up-to-date database of central and state agricultural schemes.
**Validates: Requirements 17.4**

**Property 79: Scheme Saving and Reminders**
*For any* government scheme, farmers should be able to save it to their profile and set reminders for application deadlines.
**Validates: Requirements 17.6**

**Property 80: Scheme Portal Links**
*For any* government scheme with an online application portal, the system should provide a direct link to that portal.
**Validates: Requirements 17.7**

**Property 81: Scheme Utilization Tracking**
*For any* farmer interaction with government scheme information, it should be tracked for analytics purposes.
**Validates: Requirements 17.8**

### Extension Officer Properties

**Property 82: Extension Officer Dashboard Display**
*For any* extension officer login, the dashboard should display all farmers in their assigned area with aggregated statistics.
**Validates: Requirements 18.1, 18.2**

**Property 83: Farmer Detail Information**
*For any* farmer selected by an extension officer, the system should display the farmer's fields, crop status, recent predictions, and recommendations.
**Validates: Requirements 18.3**

**Property 84: Extension Officer Notes**
*For any* farmer in an extension officer's area, the officer should be able to add notes and guidance.
**Validates: Requirements 18.4**

**Property 85: Farming Activity Reports**
*For any* extension officer request, the system should generate reports on farming activities, crop performance, and scheme adoption.
**Validates: Requirements 18.5**

**Property 86: Broadcast Messaging**
*For any* group of farmers selected by an extension officer, the officer should be able to broadcast messages to all of them.
**Validates: Requirements 18.6**

**Property 87: Interaction Tracking**
*For any* interaction between an extension officer and a farmer, it should be tracked in the system for accountability.
**Validates: Requirements 18.7**

### PWA Properties

**Property 88: PWA Standards Implementation**
*For any* user accessing the platform, it should implement PWA standards including service workers, web app manifest, and offline caching.
**Validates: Requirements 19.1**

**Property 89: PWA Installation Prompt**
*For any* first-time user visiting the platform on a supported device, the system should prompt for PWA installation.
**Validates: Requirements 19.2**

**Property 90: Lazy Loading Implementation**
*For any* images and components in the PWA, they should be lazy loaded to reduce initial load time.
**Validates: Requirements 19.5**

**Property 91: Background Sync Support**
*For any* PWA installation, it should support background sync to update data when connectivity is restored.
**Validates: Requirements 19.8**

### Analytics Properties

**Property 92: User Engagement Tracking**
*For any* user interaction with the platform, engagement metrics (feature usage, session duration) should be tracked.
**Validates: Requirements 20.1**

**Property 93: Farming Outcome Tracking**
*For any* farming outcome (actual yield, recommendation adoption), it should be tracked for effectiveness analysis.
**Validates: Requirements 20.2**

**Property 94: Monthly Report Generation**
*For any* month, the system should generate reports on platform performance, user satisfaction, and agricultural impact.
**Validates: Requirements 20.3**

**Property 95: Data Anonymization in Reports**
*For any* aggregate report generated, farmer data should be anonymized to protect privacy.
**Validates: Requirements 20.5**

**Property 96: Prediction Accuracy Tracking**
*For any* yield prediction made, accuracy should be tracked by crop type, region, and season when actual yields are reported.
**Validates: Requirements 20.6**

**Property 97: Recommendation Effectiveness Measurement**
*For any* recommendation provided, effectiveness should be measured by comparing outcomes for farmers who followed vs ignored the recommendation.
**Validates: Requirements 20.7**

**Property 98: Report Export Formats**
*For any* report generated, it should be exportable in both PDF and CSV formats.
**Validates: Requirements 20.8**

## Error Handling

### Error Classification

**Client Errors (4xx)**:
- 400 Bad Request: Invalid input data, malformed requests
- 401 Unauthorized: Missing or invalid authentication token
- 403 Forbidden: Insufficient permissions for requested resource
- 404 Not Found: Requested resource does not exist
- 429 Too Many Requests: Rate limit exceeded

**Server Errors (5xx)**:
- 500 Internal Server Error: Unexpected server-side error
- 502 Bad Gateway: Upstream service unavailable
- 503 Service Unavailable: Service temporarily down for maintenance
- 504 Gateway Timeout: Upstream service timeout

### Error Response Format

All API errors follow a standardized JSON format:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {
      "field": "specific_field",
      "reason": "validation_failed"
    },
    "timestamp": "2024-01-15T10:30:00Z",
    "requestId": "uuid"
  }
}
```

### Error Handling Strategies

**Network Errors**:
- Retry with exponential backoff (3 attempts)
- Fall back to cached data when available
- Display clear offline indicators
- Queue operations for later sync

**API Errors**:
- Log all errors with context
- Return user-friendly messages in regional language
- Provide actionable guidance (e.g., "Check your internet connection")
- Track error rates for monitoring

**ML Model Errors**:
- Fall back to ensemble average if one model fails
- Return prediction with lower confidence if data incomplete
- Log model failures for retraining consideration
- Alert administrators if error rate exceeds threshold

**Data Validation Errors**:
- Validate input on client side before submission
- Provide field-level error messages
- Suggest corrections (e.g., valid date ranges)
- Prevent submission until errors resolved

**Third-Party API Failures**:
- Implement circuit breaker pattern
- Use cached data as fallback
- Aggregate data from multiple sources
- Display staleness indicators

**Database Errors**:
- Automatic retry for transient errors
- Failover to replica databases
- Transaction rollback on failures
- Alert administrators for persistent issues

### Graceful Degradation

**Offline Mode**:
- Core features available with cached data
- Queue write operations for sync
- Clear indication of offline status
- Automatic sync when connectivity restored

**Low Bandwidth**:
- Reduce image quality
- Disable animations
- Prioritize critical data
- Use compression for API responses

**Service Degradation**:
- Disable non-critical features
- Use cached predictions if ML service down
- Display service status to users
- Maintain core functionality (view data, alerts)

## Testing Strategy

### Dual Testing Approach

The KrishiSevak platform requires both unit testing and property-based testing for comprehensive coverage:

**Unit Tests**: Verify specific examples, edge cases, and error conditions
**Property Tests**: Verify universal properties across all inputs

Both approaches are complementary and necessary. Unit tests catch concrete bugs in specific scenarios, while property tests verify general correctness across a wide range of inputs.

### Property-Based Testing

**Framework Selection**:
- **JavaScript/TypeScript**: fast-check
- **Python**: Hypothesis

**Configuration**:
- Minimum 100 iterations per property test (due to randomization)
- Each property test must reference its design document property
- Tag format: `// Feature: krishisevak-platform, Property {number}: {property_text}`

**Property Test Examples**:

```typescript
// Feature: krishisevak-platform, Property 1: User Registration Creates Account with OTP
test('user registration creates account with OTP', async () => {
  await fc.assert(
    fc.asyncProperty(
      fc.record({
        mobile: fc.string({minLength: 10, maxLength: 15}),
        name: fc.string({minLength: 2, maxLength: 100}),
        language: fc.constantFrom('hi', 'ta', 'te', 'bn', 'mr')
      }),
      async (userData) => {
        const result = await authService.register(userData);
        expect(result.userId).toBeDefined();
        expect(result.otpSent).toBe(true);
        
        const user = await db.users.findOne({mobile: userData.mobile});
        expect(user).toBeDefined();
        expect(user.otp).toHaveLength(6);
      }
    ),
    {numRuns: 100}
  );
});

// Feature: krishisevak-platform, Property 52: Rate Limiting Enforcement
test('rate limiting enforces 100 requests per minute', async () => {
  await fc.assert(
    fc.asyncProperty(
      fc.string(), // userId
      fc.integer({min: 101, max: 200}), // number of requests
      async (userId, numRequests) => {
        const requests = Array(numRequests).fill(null).map(() => 
          apiGateway.handleRequest({userId, endpoint: '/api/v1/farmers/profile'})
        );
        
        const results = await Promise.all(requests);
        const rateLimited = results.filter(r => r.status === 429);
        
        expect(rateLimited.length).toBeGreaterThan(0);
      }
    ),
    {numRuns: 100}
  );
});

// Feature: krishisevak-platform, Property 10: Low Confidence Warning Display
test('low confidence predictions show warning', async () => {
  await fc.assert(
    fc.asyncProperty(
      fc.record({
        cropType: fc.constantFrom('rice', 'wheat', 'cotton'),
        confidence: fc.float({min: 0, max: 0.69})
      }),
      async (predictionData) => {
        const prediction = {
          ...predictionData,
          predictedYield: 50,
          confidenceScore: predictionData.confidence
        };
        
        const display = await renderPrediction(prediction);
        expect(display).toContain('warning');
        expect(display).toMatch(/low confidence|कम विश्वास|குறைந்த நம்பிக்கை/);
      }
    ),
    {numRuns: 100}
  );
});
```

### Unit Testing

**Focus Areas**:
- Specific examples demonstrating correct behavior
- Edge cases (empty inputs, boundary values, special characters)
- Error conditions and exception handling
- Integration points between components

**Unit Test Examples**:

```typescript
describe('Authentication Service', () => {
  test('should reject registration with invalid mobile number', async () => {
    const result = await authService.register({
      mobile: '123', // too short
      name: 'Test User',
      language: 'hi'
    });
    
    expect(result.error).toBeDefined();
    expect(result.error.code).toBe('INVALID_MOBILE');
  });
  
  test('should handle OTP expiration correctly', async () => {
    const user = await createTestUser();
    await advanceTime(6 * 60 * 1000); // 6 minutes
    
    const result = await authService.verifyOTP(user.userId, user.otp);
    expect(result.error).toBeDefined();
    expect(result.error.code).toBe('OTP_EXPIRED');
  });
});

describe('Yield Predictor', () => {
  test('should handle missing weather data gracefully', async () => {
    const request = {
      farmerId: 'test-farmer',
      fieldId: 'test-field',
      cropType: 'rice',
      location: {latitude: 28.6139, longitude: 77.2090}
    };
    
    mockWeatherAPI.mockRejectedValue(new Error('API unavailable'));
    
    const prediction = await predictionService.predictYield(request);
    expect(prediction.confidence).toBeLessThan(0.7);
    expect(prediction.factors.weather).toBe(0);
  });
});

describe('Recommendation Engine', () => {
  test('should recommend irrigation when soil moisture is low', async () => {
    const request = {
      farmerId: 'test-farmer',
      fieldId: 'test-field',
      cropType: 'wheat',
      growthStage: 'vegetative',
      currentDate: new Date('2024-01-15')
    };
    
    mockSatelliteData({soilMoisture: 15}); // Low moisture
    
    const recommendations = await recommendationService.generate(request);
    expect(recommendations.irrigation.nextIrrigation).toBeDefined();
    expect(recommendations.irrigation.amount).toBeGreaterThan(0);
  });
});
```

### Integration Testing

**Test Scenarios**:
- End-to-end user flows (registration → field setup → prediction → recommendation)
- Service-to-service communication
- Database transactions and rollbacks
- External API integrations
- Offline-online sync scenarios

**Integration Test Example**:

```typescript
describe('Farmer Yield Prediction Flow', () => {
  test('complete flow from registration to prediction', async () => {
    // Register farmer
    const farmer = await authService.register({
      mobile: '9876543210',
      name: 'Test Farmer',
      language: 'hi'
    });
    
    // Verify OTP
    const auth = await authService.verifyOTP(farmer.userId, farmer.otp);
    
    // Register field
    const field = await farmerService.registerField({
      farmerId: farmer.userId,
      name: 'Test Field',
      area: 5.0,
      location: {latitude: 28.6139, longitude: 77.2090}
    });
    
    // Request prediction
    const prediction = await predictionService.predictYield({
      farmerId: farmer.userId,
      fieldId: field.fieldId,
      cropType: 'wheat',
      sowingDate: new Date('2023-11-01'),
      location: field.location
    });
    
    expect(prediction.predictedYield).toBeGreaterThan(0);
    expect(prediction.confidence).toBeGreaterThan(0);
    expect(prediction.harvestDate).toBeDefined();
  });
});
```

### Performance Testing

**Load Testing**:
- Simulate 10,000+ concurrent users
- Test API response times under load
- Verify database query performance
- Test ML inference latency

**Stress Testing**:
- Push system beyond normal capacity
- Identify breaking points
- Verify graceful degradation
- Test recovery procedures

**Tools**: Apache JMeter, k6, Locust

### Security Testing

**Vulnerability Scanning**:
- OWASP Top 10 vulnerabilities
- SQL injection, XSS, CSRF
- Authentication bypass attempts
- Authorization escalation

**Penetration Testing**:
- Simulated attacks on API endpoints
- Token manipulation attempts
- Rate limit bypass attempts
- Data exfiltration scenarios

**Tools**: OWASP ZAP, Burp Suite, Snyk

### Accessibility Testing

**WCAG Compliance**:
- Keyboard navigation
- Screen reader compatibility
- Color contrast ratios
- Text size and readability

**Low-Literacy Testing**:
- Icon-based navigation
- Voice interface usability
- Visual feedback clarity
- Regional language accuracy

### Continuous Integration

**CI/CD Pipeline**:
1. Code commit triggers pipeline
2. Run linters and formatters
3. Execute unit tests (must pass)
4. Execute property tests (must pass)
5. Run integration tests
6. Security scanning
7. Build Docker images
8. Deploy to staging
9. Run E2E tests on staging
10. Manual approval for production
11. Blue-green deployment to production
12. Monitor for errors

**Test Coverage Requirements**:
- Minimum 80% code coverage for unit tests
- All correctness properties must have property tests
- Critical paths must have integration tests
- All API endpoints must have security tests

### Monitoring and Observability in Testing

**Test Metrics**:
- Test execution time
- Test failure rates
- Code coverage trends
- Property test shrinking results

**Production Monitoring**:
- Real-time error tracking
- Performance metrics
- User behavior analytics
- ML model performance

**Alerting**:
- Test failures in CI/CD
- Production error rate spikes
- Performance degradation
- Security incidents

### Test Data Management

**Test Data Generation**:
- Use property-based testing generators for random data
- Maintain seed data for consistent integration tests
- Anonymize production data for testing
- Generate synthetic farming data for ML training

**Test Data Cleanup**:
- Automatic cleanup after test runs
- Isolated test databases
- Transaction rollback for unit tests
- Scheduled cleanup of staging environments

## Cost Estimation

### Initial Development Cost (70% Already Completed)

**Development Team** (₹8-10 lakhs):
- Backend developers (2): ₹4 lakhs
- Frontend developers (2): ₹3 lakhs
- ML engineer (1): ₹2 lakhs
- DevOps engineer (1): ₹1 lakh

**Cloud Infrastructure Setup** (₹1-2 lakhs):
- AWS account setup and configuration
- Kubernetes cluster setup
- Database provisioning
- CI/CD pipeline setup
- Monitoring and logging infrastructure

**Total Development Cost**: ₹9-12 lakhs (already invested)

### Pilot Deployment Cost (1000 Farmers, 3 Districts)

**Cloud Hosting - AWS** (₹3-4 lakhs annually):
- EC2 instances (t3.medium × 3): ₹1.2 lakhs/year
- RDS PostgreSQL (db.t3.medium): ₹1 lakh/year
- S3 storage (500 GB): ₹10,000/year
- CloudFront CDN: ₹30,000/year
- Elastic Load Balancer: ₹20,000/year
- ElastiCache Redis: ₹40,000/year
- Data transfer: ₹30,000/year

**API Costs** (₹2-3 lakhs annually):
- OpenWeather API (Professional plan): ₹50,000/year
- Google Earth Engine (1000 farmers × 5 fields): ₹80,000/year
- OpenAI API (GPT-4, estimated 100K queries/month): ₹1 lakh/year
- Agromonitoring API (backup, limited usage): ₹20,000/year
- Twilio SMS (10,000 SMS/month): ₹30,000/year
- Google Speech API (voice queries): ₹20,000/year

**Support & Maintenance** (₹2 lakhs annually):
- Technical support team (part-time): ₹1 lakh/year
- Bug fixes and updates: ₹50,000/year
- Security patches and monitoring: ₹50,000/year

**Training Programs** (₹1-2 lakhs one-time):
- Farmer training materials: ₹30,000
- Extension officer training: ₹50,000
- Field demonstrations: ₹70,000
- Training workshops: ₹50,000

**Total Year 1 Cost**: ₹17-23 lakhs
- Development (already invested): ₹9-12 lakhs
- Pilot deployment: ₹8-11 lakhs

**Recurring Annual Cost** (Year 2+): ₹7-9 lakhs
- Cloud hosting: ₹3-4 lakhs
- API costs: ₹2-3 lakhs
- Support & maintenance: ₹2 lakhs

### Cost Optimization Strategies

**API Cost Reduction**:
- Cache OpenAI responses for common queries (reduce by 60%)
- Use embeddings instead of completions where possible
- Implement query classification to route simple queries to rule-based system
- Batch satellite imagery requests
- Use free tier of IMD API where available

**Cloud Cost Reduction**:
- Use spot instances for non-critical workloads (save 70%)
- Implement auto-scaling to reduce idle capacity
- Use S3 Intelligent-Tiering for storage
- Optimize database queries to reduce RDS costs
- Use CloudFront caching aggressively

**Scaling Cost Projections**:

**10,000 Farmers** (₹25-35 lakhs annually):
- Cloud hosting: ₹10-12 lakhs
- API costs: ₹8-10 lakhs
- Support: ₹5 lakhs
- Training: ₹2-3 lakhs

**100,000 Farmers** (₹1.5-2 crores annually):
- Cloud hosting: ₹50-60 lakhs
- API costs: ₹40-50 lakhs
- Support team: ₹30 lakhs
- Training: ₹20-30 lakhs

### Revenue Model (Future Consideration)

**Freemium Model**:
- Basic features free for small farmers (<5 acres)
- Premium features (₹500/year): Advanced analytics, priority support, IoT integration
- Enterprise (₹5000/year): Extension officers, cooperatives, agribusinesses

**Government Partnerships**:
- Subsidized access through government schemes
- Integration with PM-KISAN and other programs
- Funding from agricultural development budgets

**Data Monetization** (with farmer consent):
- Anonymized agricultural insights for research institutions
- Crop yield forecasts for commodity markets
- Soil health trends for fertilizer companies

### Budget Allocation for Pilot (₹15-20 lakhs)

- Cloud infrastructure: 30% (₹4.5-6 lakhs)
- API costs: 25% (₹3.75-5 lakhs)
- Development completion: 20% (₹3-4 lakhs)
- Support & training: 15% (₹2.25-3 lakhs)
- Contingency: 10% (₹1.5-2 lakhs)

This budget aligns with the stated pilot budget of ₹15-20 lakhs for 1000 farmers across 3 districts.
