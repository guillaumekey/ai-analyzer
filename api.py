"""
FastAPI application for AI Visibility Audit Tool
Supports both synchronous and asynchronous processing
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
from datetime import datetime
import json
import tempfile
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import uuid
from enum import Enum
import requests

# Import your existing modules
from api_clients import OpenAIClient, PerplexityClient, GeminiClient
from utils import (
    count_brand_mentions,
    detect_competitors_from_results,
    prepare_export_data,
    export_to_json
)

# Import the API version without Streamlit
try:
    from utils.brand_llm_analysis_api import run_brand_llm_analysis
except ImportError:
    # Fallback to original if API version doesn't exist yet
    from utils.brand_llm_analysis import run_brand_llm_analysis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="AI Visibility Audit API",
    description="API for analyzing brand visibility across AI platforms",
    version="2.0.0"
)

# Configure CORS for your extranet
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your extranet URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=5)

# In-memory storage for job status (use Redis/Firestore in production)
jobs_status = {}


# Job status enum
class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# Request Models for Extranet format
class FormInfos(BaseModel):
    """Form information from extranet"""
    brand_name: str
    website: str
    prompts_list: List[str]
    prompts_count: int
    submission_date: str
    user_ip: str
    user_agent: str


class APIInfos(BaseModel):
    """API keys and credentials from extranet"""
    perplexity_api_key: Optional[str] = None
    chatgpt_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    vertex_ai: Optional[Dict[str, Any]] = None


class ExtranetRequest(BaseModel):
    """Main request model matching extranet format"""
    success: bool
    data: Dict[str, Any]  # Contains form_infos and api_infos


# Original request models (keep for backward compatibility)
class AnalysisRequest(BaseModel):
    """Request model for visibility analysis"""
    brand_name: str = Field(..., description="Brand name to analyze")
    brand_url: Optional[str] = Field(None, description="Brand website URL")
    prompts: List[str] = Field(..., description="List of prompts to test")
    competitors: Optional[List[str]] = Field([], description="List of competitor brands")
    api_keys: Dict[str, str] = Field(..., description="API keys for each platform")
    models: Optional[Dict[str, str]] = Field({}, description="Model selection for each platform")
    include_llm_analysis: bool = Field(False, description="Include brand LLM analysis")
    language: str = Field("en", description="Language for analysis (en/fr)")


# Response Models
class AnalysisResponse(BaseModel):
    """Response model for analysis results"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    analysis_id: Optional[str] = None
    processing_time: Optional[float] = None


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint for Cloud Run"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "endpoints": {
            "sync": "/analyze",
            "async": "/analyze-async"
        }
    }


# Main SYNCHRONOUS analysis endpoint
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_brand_visibility(
        request: Request,
        background_tasks: BackgroundTasks
):
    """
    Analyze brand visibility across AI platforms (SYNCHRONOUS)
    Accepts both extranet format and original format
    """
    start_time = datetime.now()

    try:
        # Get raw JSON data
        raw_data = await request.json()
        logger.info(f"Received request: {json.dumps(raw_data, indent=2)[:500]}...")

        # Check if it's extranet format or original format
        if 'success' in raw_data and 'data' in raw_data:
            # Extranet format
            logger.info("Processing extranet format request")

            # Extract data
            form_infos = raw_data['data'].get('form_infos', {})
            api_infos = raw_data['data'].get('api_infos', {})

            # Transform to internal format
            brand_name = form_infos.get('brand_name', '')
            brand_url = form_infos.get('website', '')
            prompts = form_infos.get('prompts_list', [])

            # Map API keys
            api_keys = {
                'openai': api_infos.get('chatgpt_api_key', ''),
                'perplexity': api_infos.get('perplexity_api_key', ''),
                'gemini': api_infos.get('gemini_api_key', '')
            }

            # Handle Vertex AI credentials if provided
            vertex_credentials_path = None
            if api_infos.get('vertex_ai'):
                try:
                    # Save Vertex AI credentials temporarily
                    with tempfile.NamedTemporaryFile(
                            mode='w',
                            suffix='.json',
                            delete=False
                    ) as tmp_file:
                        json.dump(api_infos['vertex_ai'], tmp_file)
                        vertex_credentials_path = tmp_file.name
                        logger.info(f"Saved Vertex AI credentials to {vertex_credentials_path}")
                except Exception as e:
                    logger.error(f"Error saving Vertex AI credentials: {str(e)}")

            # Default values
            competitors = []
            include_llm_analysis = True  # Always include for extranet
            language = 'fr'  # Default to French for extranet
            models = {
                'openai': 'gpt-3.5-turbo',
                'perplexity': 'sonar',
                'gemini': 'gemini-1.5-flash'
            }

        else:
            # Original format
            logger.info("Processing original format request")
            analysis_request = AnalysisRequest(**raw_data)

            brand_name = analysis_request.brand_name
            brand_url = analysis_request.brand_url
            prompts = analysis_request.prompts
            competitors = analysis_request.competitors or []
            api_keys = analysis_request.api_keys
            models = analysis_request.models or {}
            include_llm_analysis = analysis_request.include_llm_analysis
            language = analysis_request.language
            vertex_credentials_path = None

        # Validate required fields
        if not brand_name:
            raise HTTPException(status_code=400, detail="Brand name is required")
        if not prompts:
            raise HTTPException(status_code=400, detail="At least one prompt is required")
        if not any(api_keys.values()):
            raise HTTPException(status_code=400, detail="At least one API key is required")

        # Initialize clients
        clients = {}
        selected_models = models

        if api_keys.get('openai'):
            clients['chatgpt'] = OpenAIClient(
                api_keys['openai'],
                selected_models.get('openai', 'gpt-3.5-turbo')
            )
            logger.info("Initialized OpenAI client")

        if api_keys.get('perplexity'):
            clients['perplexity'] = PerplexityClient(
                api_keys['perplexity'],
                selected_models.get('perplexity', 'sonar')
            )
            logger.info("Initialized Perplexity client")

        if api_keys.get('gemini'):
            clients['gemini'] = GeminiClient(
                api_keys['gemini'],
                selected_models.get('gemini', 'gemini-1.5-flash')
            )
            logger.info("Initialized Gemini client")

        # Process prompts
        logger.info(f"Processing {len(prompts)} prompts for brand: {brand_name}")
        results = await process_prompts_async(
            prompts,
            brand_name,
            clients
        )

        # Auto-detect competitors if OpenAI is available
        all_competitors = competitors
        if api_keys.get('openai') and clients.get('chatgpt'):
            logger.info("Detecting competitors...")
            detection_results = detect_competitors_from_results(
                clients['chatgpt'],
                results,
                brand_name,
                all_competitors
            )
            detected = detection_results.get('detected', [])
            all_competitors = list(set(all_competitors + detected))
            logger.info(f"Detected competitors: {detected}")

        # Add competitor tracking
        if all_competitors:
            results = add_competitor_tracking(results, all_competitors)

        # Calculate totals
        active_platforms = list(results.keys())
        total_unique = sum(results[p]['unique_mentions'] for p in active_platforms)
        total_mentions = sum(results[p]['total_mentions'] for p in active_platforms)
        total_queries = len(prompts) * len(active_platforms)

        # Brand LLM analysis if requested
        brand_analysis = None
        if include_llm_analysis and clients:
            logger.info("Running brand LLM analysis...")
            brand_analysis = await run_brand_llm_analysis_async(
                brand_name,
                clients,
                selected_models,
                vertex_credentials_path
            )

        # Prepare export data
        export_data = prepare_export_data(
            brand_name=brand_name,
            brand_url=brand_url,
            prompts=prompts,
            results=results,
            unique_mentions=total_unique,
            total_mentions=total_mentions,
            total_queries=total_queries,
            competitors=all_competitors,
            brand_analysis=brand_analysis
        )

        # Clean up temporary Vertex AI file
        if vertex_credentials_path and os.path.exists(vertex_credentials_path):
            try:
                os.remove(vertex_credentials_path)
                logger.info("Cleaned up temporary Vertex AI credentials")
            except:
                pass

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Analysis completed in {processing_time:.2f} seconds")

        return AnalysisResponse(
            success=True,
            message="Analysis completed successfully",
            data=export_data,
            analysis_id=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            processing_time=processing_time
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        return AnalysisResponse(
            success=False,
            message=f"Analysis failed: {str(e)}",
            data=None,
            processing_time=(datetime.now() - start_time).total_seconds()
        )


# ASYNCHRONOUS analysis endpoint
@app.post("/analyze-async")
async def start_async_analysis(
        request: Request,
        background_tasks: BackgroundTasks
):
    """
    Start async analysis and return job ID immediately
    """
    try:
        # Parse request
        raw_data = await request.json()

        # Generate unique job ID
        job_id = f"job_{uuid.uuid4().hex[:12]}"

        # Extract webhook URL
        webhook_url = raw_data.get('webhook_url')
        if not webhook_url:
            raise HTTPException(status_code=400, detail="webhook_url is required for async analysis")

        # Initialize job status
        jobs_status[job_id] = {
            "status": JobStatus.PENDING,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "progress": 0,
            "message": "Analysis queued",
            "webhook_url": webhook_url,
            "raw_data": raw_data  # Store for processing
        }

        # Add background task
        background_tasks.add_task(
            process_analysis_async,
            job_id
        )

        # Return job ID immediately
        return {
            "success": True,
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "message": "Analysis started. You will be notified via webhook when complete.",
            "status_url": f"/job/{job_id}"
        }

    except Exception as e:
        logger.error(f"Error starting async analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Get job status endpoint
@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """
    Get current job status
    """
    if job_id not in jobs_status:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs_status[job_id]

    return {
        "job_id": job_id,
        "status": job["status"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "progress": job.get("progress"),
        "message": job.get("message"),
        "error": job.get("error") if job["status"] == JobStatus.FAILED else None
    }


# Get job result endpoint
@app.get("/job/{job_id}/result")
async def get_job_result(job_id: str):
    """
    Get job result (only if completed)
    """
    if job_id not in jobs_status:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs_status[job_id]

    if job["status"] != JobStatus.COMPLETED:
        return {
            "success": False,
            "message": f"Job is {job['status']}. Results not available yet.",
            "status": job["status"]
        }

    return {
        "success": True,
        "job_id": job_id,
        "data": job.get("result"),
        "processing_time": job.get("processing_time")
    }


# Background processing function
async def process_analysis_async(job_id: str):
    """
    Process analysis in background and call webhook when done
    """
    start_time = datetime.now()

    try:
        # Get stored data
        job_data = jobs_status[job_id]
        raw_data = job_data["raw_data"]

        # Update status
        jobs_status[job_id].update({
            "status": JobStatus.PROCESSING,
            "updated_at": datetime.now().isoformat(),
            "progress": 10,
            "message": "Initializing analysis..."
        })

        # Extract data (same as synchronous endpoint)
        form_infos = raw_data['data'].get('form_infos', {})
        api_infos = raw_data['data'].get('api_infos', {})

        brand_name = form_infos.get('brand_name', '')
        brand_url = form_infos.get('website', '')
        prompts = form_infos.get('prompts_list', [])

        # Map API keys
        api_keys = {
            'openai': api_infos.get('chatgpt_api_key', ''),
            'perplexity': api_infos.get('perplexity_api_key', ''),
            'gemini': api_infos.get('gemini_api_key', '')
        }

        # Update progress
        jobs_status[job_id].update({
            "progress": 20,
            "message": "Setting up API clients..."
        })

        # Initialize clients
        clients = {}
        if api_keys.get('openai'):
            clients['chatgpt'] = OpenAIClient(api_keys['openai'], 'gpt-3.5-turbo')
        if api_keys.get('perplexity'):
            clients['perplexity'] = PerplexityClient(api_keys['perplexity'], 'sonar')
        if api_keys.get('gemini'):
            clients['gemini'] = GeminiClient(api_keys['gemini'], 'gemini-1.5-flash')

        # Process prompts with progress updates
        jobs_status[job_id].update({
            "progress": 30,
            "message": f"Processing {len(prompts)} prompts..."
        })

        results = await process_prompts_with_progress(
            prompts, brand_name, clients, job_id
        )

        # Detect competitors
        jobs_status[job_id].update({
            "progress": 70,
            "message": "Detecting competitors..."
        })

        all_competitors = []
        if api_keys.get('openai') and clients.get('chatgpt'):
            detection_results = detect_competitors_from_results(
                clients['chatgpt'], results, brand_name, []
            )
            all_competitors = detection_results.get('detected', [])

        # Add competitor tracking
        if all_competitors:
            results = add_competitor_tracking(results, all_competitors)

        # Calculate totals
        active_platforms = list(results.keys())
        total_unique = sum(results[p]['unique_mentions'] for p in active_platforms)
        total_mentions = sum(results[p]['total_mentions'] for p in active_platforms)
        total_queries = len(prompts) * len(active_platforms)

        # Brand LLM analysis
        jobs_status[job_id].update({
            "progress": 80,
            "message": "Running brand analysis..."
        })

        brand_analysis = None
        if clients:
            brand_analysis = await run_brand_llm_analysis_async(
                brand_name, clients, {'openai': 'gpt-3.5-turbo'}
            )

        # Prepare final data
        jobs_status[job_id].update({
            "progress": 90,
            "message": "Preparing results..."
        })

        export_data = prepare_export_data(
            brand_name=brand_name,
            brand_url=brand_url,
            prompts=prompts,
            results=results,
            unique_mentions=total_unique,
            total_mentions=total_mentions,
            total_queries=total_queries,
            competitors=all_competitors,
            brand_analysis=brand_analysis
        )

        processing_time = (datetime.now() - start_time).total_seconds()

        # Update job as completed
        jobs_status[job_id].update({
            "status": JobStatus.COMPLETED,
            "updated_at": datetime.now().isoformat(),
            "progress": 100,
            "message": "Analysis completed successfully",
            "result": export_data,
            "processing_time": processing_time
        })

        # Call webhook
        webhook_url = jobs_status[job_id]["webhook_url"]
        await notify_webhook(webhook_url, job_id, True, export_data, processing_time)

    except Exception as e:
        logger.error(f"Error in async processing: {str(e)}")

        # Update job as failed
        jobs_status[job_id].update({
            "status": JobStatus.FAILED,
            "updated_at": datetime.now().isoformat(),
            "message": "Analysis failed",
            "error": str(e)
        })

        # Notify webhook of failure
        webhook_url = jobs_status[job_id]["webhook_url"]
        await notify_webhook(webhook_url, job_id, False, None, None, str(e))


# Process prompts with progress updates
async def process_prompts_with_progress(prompts, brand_name, clients, job_id):
    """Process prompts and update progress"""
    results = {
        platform: {
            'responses': [],
            'total_mentions': 0,
            'unique_mentions': 0
        }
        for platform in clients.keys()
    }

    total_steps = len(prompts) * len(clients)
    current_step = 0

    for i, prompt in enumerate(prompts):
        for platform_key, client in clients.items():
            try:
                if platform_key == 'perplexity':
                    response, sources = client.call_api(prompt)
                    mentions = count_brand_mentions(response, brand_name)

                    results[platform_key]['responses'].append({
                        'prompt': prompt,
                        'response': response,
                        'mentions': mentions,
                        'sources': sources
                    })
                else:
                    response = client.call_api(prompt)
                    mentions = count_brand_mentions(response, brand_name)

                    results[platform_key]['responses'].append({
                        'prompt': prompt,
                        'response': response,
                        'mentions': mentions
                    })

                results[platform_key]['total_mentions'] += mentions
                if mentions > 0:
                    results[platform_key]['unique_mentions'] += 1

            except Exception as e:
                logger.error(f"Error processing {platform_key}: {str(e)}")
                results[platform_key]['responses'].append({
                    'prompt': prompt,
                    'response': f"Error: {str(e)}",
                    'mentions': 0
                })

            # Update progress
            current_step += 1
            progress = 30 + int((current_step / total_steps) * 40)  # 30-70% range
            jobs_status[job_id]["progress"] = progress
            jobs_status[job_id]["message"] = f"Processing prompt {i + 1}/{len(prompts)} on {platform_key}..."

    return results


# Notify webhook
async def notify_webhook(webhook_url: str, job_id: str, success: bool,
                         data: Any, processing_time: float = None, error: str = None):
    """Send webhook notification"""
    try:
        payload = {
            "job_id": job_id,
            "success": success,
            "timestamp": datetime.now().isoformat(),
        }

        if success:
            payload.update({
                "data": data,
                "processing_time": processing_time,
                "result_url": f"/job/{job_id}/result"
            })
        else:
            payload["error"] = error

        # Send webhook
        response = requests.post(
            webhook_url,
            json=payload,
            timeout=30,
            headers={'Content-Type': 'application/json'}
        )

        logger.info(f"Webhook sent to {webhook_url}: {response.status_code}")

    except Exception as e:
        logger.error(f"Failed to send webhook: {str(e)}")


# Async wrapper for prompt processing
async def process_prompts_async(prompts, brand_name, clients):
    """Process prompts asynchronously"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        process_prompts_sync,
        prompts,
        brand_name,
        clients
    )


def process_prompts_sync(prompts, brand_name, clients):
    """Synchronous prompt processing (reusing existing logic)"""
    results = {
        platform: {
            'responses': [],
            'total_mentions': 0,
            'unique_mentions': 0
        }
        for platform in clients.keys()
    }

    for prompt in prompts:
        for platform_key, client in clients.items():
            try:
                if platform_key == 'perplexity':
                    response, sources = client.call_api(prompt)
                    mentions = count_brand_mentions(response, brand_name)

                    results[platform_key]['responses'].append({
                        'prompt': prompt,
                        'response': response,
                        'mentions': mentions,
                        'sources': sources
                    })
                else:
                    response = client.call_api(prompt)
                    mentions = count_brand_mentions(response, brand_name)

                    results[platform_key]['responses'].append({
                        'prompt': prompt,
                        'response': response,
                        'mentions': mentions
                    })

                results[platform_key]['total_mentions'] += mentions
                if mentions > 0:
                    results[platform_key]['unique_mentions'] += 1

            except Exception as e:
                logger.error(f"Error processing {platform_key} for prompt '{prompt}': {str(e)}")
                results[platform_key]['responses'].append({
                    'prompt': prompt,
                    'response': f"Error: {str(e)}",
                    'mentions': 0
                })

    return results


def add_competitor_tracking(results, competitors):
    """Add competitor tracking to results"""
    for platform_key, platform_data in results.items():
        platform_data['competitor_mentions'] = {
            comp: {'total': 0, 'unique': 0} for comp in competitors
        }

        for response_data in platform_data['responses']:
            response_text = response_data.get('response', '')
            response_data['competitor_mentions'] = {}

            for competitor in competitors:
                mentions = count_brand_mentions(response_text, competitor)
                response_data['competitor_mentions'][competitor] = mentions

                platform_data['competitor_mentions'][competitor]['total'] += mentions
                if mentions > 0:
                    platform_data['competitor_mentions'][competitor]['unique'] += 1

    return results


# Async wrapper for LLM analysis with Vertex support
async def run_brand_llm_analysis_async(brand_name, clients, models, vertex_credentials_path=None):
    """Run brand LLM analysis asynchronously with optional Vertex AI"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        run_brand_llm_analysis,
        brand_name,
        clients,
        models,
        vertex_credentials_path
    )


# Simple test endpoint
@app.post("/test")
async def test_endpoint(request: Request):
    """Test endpoint to verify request format"""
    try:
        data = await request.json()
        return {
            "success": True,
            "message": "Request received successfully",
            "received_data": data
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error: {str(e)}"
        }


if __name__ == "__main__":
    import uvicorn

    # For local testing
    uvicorn.run(app, host="0.0.0.0", port=8080)