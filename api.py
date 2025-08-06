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

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
        logger.info("=" * 80)
        logger.info("🚀 NOUVELLE REQUÊTE SYNCHRONE REÇUE")
        logger.info(f"📅 Timestamp: {datetime.now().isoformat()}")
        logger.info(f"📦 Taille des données reçues: {len(json.dumps(raw_data))} caractères")

        # Log structure of received data
        logger.info("📋 Structure des données reçues:")
        logger.info(f"  - Clés principales: {list(raw_data.keys())}")
        if 'data' in raw_data:
            logger.info(f"  - Clés dans 'data': {list(raw_data['data'].keys())}")

        # Check if it's extranet format or original format
        if 'success' in raw_data and 'data' in raw_data:
            # Extranet format
            logger.info("✅ Format EXTRANET détecté")

            # Extract data
            form_infos = raw_data['data'].get('form_infos', {})
            api_infos = raw_data['data'].get('api_infos', {})

            logger.info("📝 FORM INFOS:")
            logger.info(f"  - Brand: {form_infos.get('brand_name', 'N/A')}")
            logger.info(f"  - Website: {form_infos.get('website', 'N/A')}")
            logger.info(f"  - Nombre de prompts: {len(form_infos.get('prompts_list', []))}")
            if form_infos.get('prompts_list'):
                logger.info(f"  - Premier prompt: {form_infos['prompts_list'][0][:50]}...")

            # Transform to internal format
            brand_name = form_infos.get('brand_name', '')
            brand_url = form_infos.get('website', '')
            prompts = form_infos.get('prompts_list', [])

            # Log API keys info (sans exposer les clés)
            logger.info("🔑 API INFOS:")
            logger.info(f"  - ChatGPT key présente: {bool(api_infos.get('chatgpt_api_key'))}")
            logger.info(f"  - ChatGPT key longueur: {len(api_infos.get('chatgpt_api_key', ''))}")
            logger.info(f"  - Perplexity key présente: {bool(api_infos.get('perplexity_api_key'))}")
            logger.info(f"  - Perplexity key longueur: {len(api_infos.get('perplexity_api_key', ''))}")
            logger.info(f"  - Gemini key présente: {bool(api_infos.get('gemini_api_key'))}")
            logger.info(f"  - Gemini key longueur: {len(api_infos.get('gemini_api_key', ''))}")
            logger.info(f"  - Vertex AI présent: {bool(api_infos.get('vertex_ai'))}")

            # Map API keys
            api_keys = {
                'openai': api_infos.get('chatgpt_api_key', ''),
                'perplexity': api_infos.get('perplexity_api_key', ''),
                'gemini': api_infos.get('gemini_api_key', '')
            }

            logger.info("🔄 MAPPING DES CLÉS API:")
            logger.info(
                f"  - OpenAI: {'✅ Présente' if api_keys['openai'] else '❌ Absente'} (longueur: {len(api_keys['openai'])})")
            logger.info(
                f"  - Perplexity: {'✅ Présente' if api_keys['perplexity'] else '❌ Absente'} (longueur: {len(api_keys['perplexity'])})")
            logger.info(
                f"  - Gemini: {'✅ Présente' if api_keys['gemini'] else '❌ Absente'} (longueur: {len(api_keys['gemini'])})")

            # Handle Vertex AI credentials if provided
            vertex_credentials_path = None
            if api_infos.get('vertex_ai'):
                try:
                    logger.info("☁️ Traitement des credentials Vertex AI...")
                    # Save Vertex AI credentials temporarily
                    with tempfile.NamedTemporaryFile(
                            mode='w',
                            suffix='.json',
                            delete=False
                    ) as tmp_file:
                        json.dump(api_infos['vertex_ai'], tmp_file)
                        vertex_credentials_path = tmp_file.name
                        logger.info(f"✅ Vertex AI credentials sauvegardées: {vertex_credentials_path}")
                except Exception as e:
                    logger.error(f"❌ Erreur Vertex AI credentials: {str(e)}")

            # Default values
            competitors = []
            include_llm_analysis = True  # Always include for extranet
            language = 'fr'  # Default to French for extranet
            models = {
                'openai': 'gpt-3.5-turbo',
                'perplexity': 'sonar',
                'gemini': 'gemini-1.5-flash'
            }
            logger.info(f"🌐 Langue définie: {language}")
            logger.info(f"🤖 Modèles sélectionnés: {models}")

        else:
            # Original format
            logger.info("📋 Format ORIGINAL détecté")
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
        logger.info("✔️ VALIDATION DES DONNÉES:")
        if not brand_name:
            logger.error("❌ Nom de marque manquant")
            raise HTTPException(status_code=400, detail="Brand name is required")
        logger.info(f"  - Brand name: ✅ '{brand_name}'")

        if not prompts:
            logger.error("❌ Aucun prompt fourni")
            raise HTTPException(status_code=400, detail="At least one prompt is required")
        logger.info(f"  - Prompts: ✅ {len(prompts)} prompts")

        if not any(api_keys.values()):
            logger.error("❌ Aucune clé API fournie")
            raise HTTPException(status_code=400, detail="At least one API key is required")
        logger.info(f"  - API keys: ✅ Au moins une clé présente")

        # Initialize clients
        logger.info("🏗️ INITIALISATION DES CLIENTS API:")
        clients = {}
        selected_models = models

        if api_keys.get('openai'):
            try:
                logger.info(f"  📌 Création client OpenAI avec modèle: {selected_models.get('openai', 'gpt-3.5-turbo')}")
                clients['chatgpt'] = OpenAIClient(
                    api_keys['openai'],
                    selected_models.get('openai', 'gpt-3.5-turbo')
                )
                logger.info("  ✅ Client OpenAI créé avec succès")
            except Exception as e:
                logger.error(f"  ❌ Erreur création client OpenAI: {str(e)}")
        else:
            logger.warning("  ⚠️ Pas de clé OpenAI - client non créé")

        if api_keys.get('perplexity'):
            try:
                logger.info(f"  📌 Création client Perplexity avec modèle: {selected_models.get('perplexity', 'sonar')}")
                clients['perplexity'] = PerplexityClient(
                    api_keys['perplexity'],
                    selected_models.get('perplexity', 'sonar')
                )
                logger.info("  ✅ Client Perplexity créé avec succès")
            except Exception as e:
                logger.error(f"  ❌ Erreur création client Perplexity: {str(e)}")
        else:
            logger.warning("  ⚠️ Pas de clé Perplexity - client non créé")

        if api_keys.get('gemini'):
            try:
                logger.info(
                    f"  📌 Création client Gemini avec modèle: {selected_models.get('gemini', 'gemini-1.5-flash')}")
                clients['gemini'] = GeminiClient(
                    api_keys['gemini'],
                    selected_models.get('gemini', 'gemini-1.5-flash')
                )
                logger.info("  ✅ Client Gemini créé avec succès")
            except Exception as e:
                logger.error(f"  ❌ Erreur création client Gemini: {str(e)}")
        else:
            logger.warning("  ⚠️ Pas de clé Gemini - client non créé")

        logger.info(f"📊 RÉSUMÉ DES CLIENTS:")
        logger.info(f"  - Nombre total de clients créés: {len(clients)}")
        logger.info(f"  - Plateformes actives: {list(clients.keys())}")

        # Process prompts
        logger.info(f"🔄 DÉBUT DU TRAITEMENT DES PROMPTS")
        logger.info(f"  - Nombre de prompts: {len(prompts)}")
        logger.info(f"  - Nombre de plateformes: {len(clients)}")
        logger.info(f"  - Total de requêtes à faire: {len(prompts) * len(clients)}")

        results = await process_prompts_async(
            prompts,
            brand_name,
            clients
        )

        # Auto-detect competitors if OpenAI is available
        all_competitors = competitors
        if api_keys.get('openai') and clients.get('chatgpt'):
            logger.info("🔍 DÉTECTION AUTOMATIQUE DES CONCURRENTS...")
            detection_results = detect_competitors_from_results(
                clients['chatgpt'],
                results,
                brand_name,
                all_competitors
            )
            detected = detection_results.get('detected', [])
            all_competitors = list(set(all_competitors + detected))
            logger.info(f"  - Concurrents détectés: {detected}")
            logger.info(f"  - Total concurrents: {len(all_competitors)}")
        else:
            logger.info("⏭️ Détection des concurrents ignorée (pas de client OpenAI)")

        # Add competitor tracking
        if all_competitors:
            logger.info(f"📊 Ajout du tracking des concurrents pour {len(all_competitors)} marques")
            results = add_competitor_tracking(results, all_competitors)

        # Calculate totals
        active_platforms = list(results.keys())
        total_unique = sum(results[p]['unique_mentions'] for p in active_platforms)
        total_mentions = sum(results[p]['total_mentions'] for p in active_platforms)
        total_queries = len(prompts) * len(active_platforms)

        logger.info("📈 STATISTIQUES FINALES:")
        logger.info(f"  - Mentions uniques: {total_unique}")
        logger.info(f"  - Mentions totales: {total_mentions}")
        logger.info(f"  - Requêtes totales: {total_queries}")

        # Brand LLM analysis if requested
        brand_analysis = None
        if include_llm_analysis and clients:
            logger.info("🧠 DÉBUT DE L'ANALYSE LLM DE LA MARQUE...")
            brand_analysis = await run_brand_llm_analysis_async(
                brand_name,
                clients,
                selected_models,
                vertex_credentials_path
            )
            logger.info("✅ Analyse LLM terminée")
        else:
            logger.info("⏭️ Analyse LLM ignorée")

        # Prepare export data
        logger.info("💾 Préparation des données d'export...")
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
                logger.info("🗑️ Fichier Vertex AI temporaire supprimé")
            except:
                pass

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info("=" * 80)
        logger.info(f"✅ ANALYSE TERMINÉE AVEC SUCCÈS")
        logger.info(f"⏱️ Temps de traitement: {processing_time:.2f} secondes")
        logger.info("=" * 80)

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
        logger.error("=" * 80)
        logger.error(f"❌ ERREUR DANS L'ANALYSE: {str(e)}")
        logger.error(f"Type d'erreur: {type(e).__name__}")
        logger.error("Traceback complet:", exc_info=True)
        logger.error("=" * 80)
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
        logger.info("=" * 80)
        logger.info("🚀 NOUVELLE REQUÊTE ASYNCHRONE REÇUE")
        logger.info(f"📅 Timestamp: {datetime.now().isoformat()}")

        # Generate unique job ID
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        logger.info(f"🆔 Job ID généré: {job_id}")

        # Extract webhook URL
        webhook_url = raw_data.get('webhook_url')
        if not webhook_url:
            logger.error("❌ webhook_url manquant dans la requête")
            raise HTTPException(status_code=400, detail="webhook_url is required for async analysis")

        logger.info(f"🔗 Webhook URL: {webhook_url}")

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

        logger.info(f"✅ Job {job_id} ajouté à la queue")
        logger.info("=" * 80)

        # Return job ID immediately
        return {
            "success": True,
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "message": "Analysis started. You will be notified via webhook when complete.",
            "status_url": f"/job/{job_id}"
        }

    except Exception as e:
        logger.error(f"❌ Erreur lors du démarrage de l'analyse async: {str(e)}")
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

    logger.info("=" * 80)
    logger.info(f"🔄 DÉBUT DU TRAITEMENT ASYNCHRONE - Job: {job_id}")
    logger.info(f"📅 Démarré à: {start_time.isoformat()}")

    try:
        # Get stored data
        job_data = jobs_status[job_id]
        raw_data = job_data["raw_data"]

        logger.info("📦 Récupération des données du job...")

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

        logger.info("📝 DONNÉES EXTRAITES:")
        logger.info(f"  - Brand: {form_infos.get('brand_name', 'N/A')}")
        logger.info(f"  - Nombre de prompts: {len(form_infos.get('prompts_list', []))}")

        brand_name = form_infos.get('brand_name', '')
        brand_url = form_infos.get('website', '')
        prompts = form_infos.get('prompts_list', [])

        # Map API keys
        api_keys = {
            'openai': api_infos.get('chatgpt_api_key', ''),
            'perplexity': api_infos.get('perplexity_api_key', ''),
            'gemini': api_infos.get('gemini_api_key', '')
        }

        logger.info("🔑 CLÉS API MAPPÉES:")
        logger.info(f"  - OpenAI: {'✅' if api_keys['openai'] else '❌'}")
        logger.info(f"  - Perplexity: {'✅' if api_keys['perplexity'] else '❌'}")
        logger.info(f"  - Gemini: {'✅' if api_keys['gemini'] else '❌'}")

        # Update progress
        jobs_status[job_id].update({
            "progress": 20,
            "message": "Setting up API clients..."
        })

        # Initialize clients
        logger.info("🏗️ CRÉATION DES CLIENTS API...")
        clients = {}

        if api_keys.get('openai'):
            try:
                clients['chatgpt'] = OpenAIClient(api_keys['openai'], 'gpt-3.5-turbo')
                logger.info("  ✅ Client OpenAI créé")
            except Exception as e:
                logger.error(f"  ❌ Erreur client OpenAI: {str(e)}")

        if api_keys.get('perplexity'):
            try:
                clients['perplexity'] = PerplexityClient(api_keys['perplexity'], 'sonar')
                logger.info("  ✅ Client Perplexity créé")
            except Exception as e:
                logger.error(f"  ❌ Erreur client Perplexity: {str(e)}")

        if api_keys.get('gemini'):
            try:
                clients['gemini'] = GeminiClient(api_keys['gemini'], 'gemini-1.5-flash')
                logger.info("  ✅ Client Gemini créé")
            except Exception as e:
                logger.error(f"  ❌ Erreur client Gemini: {str(e)}")

        logger.info(f"📊 Total clients créés: {len(clients)} - {list(clients.keys())}")

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
            logger.info("🔍 Détection des concurrents...")
            detection_results = detect_competitors_from_results(
                clients['chatgpt'], results, brand_name, []
            )
            all_competitors = detection_results.get('detected', [])
            logger.info(f"  - Concurrents trouvés: {all_competitors}")

        # Add competitor tracking
        if all_competitors:
            results = add_competitor_tracking(results, all_competitors)

        # Calculate totals
        active_platforms = list(results.keys())
        total_unique = sum(results[p]['unique_mentions'] for p in active_platforms)
        total_mentions = sum(results[p]['total_mentions'] for p in active_platforms)
        total_queries = len(prompts) * len(active_platforms)

        logger.info("📈 RÉSULTATS:")
        logger.info(f"  - Mentions uniques: {total_unique}")
        logger.info(f"  - Mentions totales: {total_mentions}")

        # Brand LLM analysis
        jobs_status[job_id].update({
            "progress": 80,
            "message": "Running brand analysis..."
        })

        brand_analysis = None
        if clients:
            logger.info("🧠 Analyse LLM de la marque...")
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

        logger.info("=" * 80)
        logger.info(f"✅ JOB {job_id} TERMINÉ AVEC SUCCÈS")
        logger.info(f"⏱️ Temps total: {processing_time:.2f} secondes")
        logger.info(f"🔔 Envoi du webhook à: {jobs_status[job_id]['webhook_url']}")
        logger.info("=" * 80)

        # Call webhook
        webhook_url = jobs_status[job_id]["webhook_url"]
        await notify_webhook(webhook_url, job_id, True, export_data, processing_time)

    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"❌ ERREUR DANS LE JOB {job_id}: {str(e)}")
        logger.error(f"Type: {type(e).__name__}")
        logger.error("Traceback:", exc_info=True)
        logger.error("=" * 80)

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
    logger.info(f"🔄 Début du traitement de {len(prompts)} prompts sur {len(clients)} plateformes")

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
        logger.info(f"  📝 Traitement prompt {i + 1}/{len(prompts)}: {prompt[:50]}...")

        for platform_key, client in clients.items():
            try:
                logger.info(f"    🌐 Appel API {platform_key}...")

                if platform_key == 'perplexity':
                    response, sources = client.call_api(prompt)
                    mentions = count_brand_mentions(response, brand_name)
                    logger.info(f"    ✅ Réponse reçue - Mentions: {mentions}, Sources: {len(sources)}")

                    results[platform_key]['responses'].append({
                        'prompt': prompt,
                        'response': response,
                        'mentions': mentions,
                        'sources': sources
                    })
                else:
                    response = client.call_api(prompt)
                    mentions = count_brand_mentions(response, brand_name)
                    logger.info(f"    ✅ Réponse reçue - Mentions: {mentions}")

                    results[platform_key]['responses'].append({
                        'prompt': prompt,
                        'response': response,
                        'mentions': mentions
                    })

                results[platform_key]['total_mentions'] += mentions
                if mentions > 0:
                    results[platform_key]['unique_mentions'] += 1

            except Exception as e:
                logger.error(f"    ❌ Erreur {platform_key}: {str(e)}")
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

    logger.info(f"✅ Traitement des prompts terminé")
    return results

# Notify webhook
async def notify_webhook(webhook_url: str, job_id: str, success: bool,
                         data: Any, processing_time: float = None, error: str = None):
    """Send webhook notification"""
    try:
        logger.info(f"📮 Envoi du webhook...")
        logger.info(f"  - URL: {webhook_url}")
        logger.info(f"  - Job ID: {job_id}")
        logger.info(f"  - Success: {success}")

        # Préparer le payload
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
            logger.info(f"  - Processing time: {processing_time:.2f}s")
            logger.info(f"  - Data keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}")
        else:
            payload["error"] = error
            logger.info(f"  - Error: {error}")

        # Headers avec User-Agent pour l'extranet
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'AI-Visibility-API/1.0 (Webhook)',
            'X-Webhook-Source': 'ai-visibility-api',
            'X-Job-ID': job_id
        }

        # Log du payload pour debug
        payload_json = json.dumps(payload)
        logger.info(f"📤 Payload size: {len(payload_json)} bytes")
        logger.info(f"📤 First 500 chars of payload: {payload_json[:500]}...")

        # Si le payload est très gros, logger plus d'infos
        if len(payload_json) > 10000:
            logger.warning(f"⚠️ Large payload detected: {len(payload_json)} bytes")

        # Send webhook avec session pour réutiliser la connexion
        import aiohttp
        async with aiohttp.ClientSession() as session:
            try:
                logger.info("🔌 Envoi de la requête HTTP...")

                async with session.post(
                        webhook_url,
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=60)
                ) as response:

                    response_text = await response.text()

                    logger.info(f"✅ Webhook envoyé - Status: {response.status}")
                    logger.info(f"📥 Response headers: {dict(response.headers)}")
                    logger.info(f"📥 Response body: {response_text[:500]}...")

                    if response.status != 200:
                        logger.warning(f"⚠️ Réponse webhook non-200: {response_text}")

                    # S'assurer que la réponse est bien reçue
                    await response.read()

            except asyncio.TimeoutError:
                logger.error("❌ Timeout lors de l'envoi du webhook (60s)")
            except aiohttp.ClientConnectorError as e:
                logger.error(f"❌ Erreur de connexion: {str(e)}")
                logger.error(f"   - URL: {webhook_url}")
                logger.error(f"   - Type: {type(e).__name__}")
            except aiohttp.ClientError as e:
                logger.error(f"❌ Erreur client HTTP: {str(e)}")
            except Exception as e:
                logger.error(f"❌ Erreur inattendue lors de l'envoi: {str(e)}")
                logger.error(f"   - Type: {type(e).__name__}")

        # Alternative avec requests si aiohttp échoue
        try:
            logger.info("🔄 Tentative avec requests (synchrone)...")

            response = requests.post(
                webhook_url,
                json=payload,
                headers=headers,
                timeout=60,
                verify=True  # Vérifier SSL
            )

            logger.info(f"✅ Webhook envoyé (requests) - Status: {response.status_code}")
            logger.info(f"📥 Response: {response.text[:500]}...")

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Échec aussi avec requests: {str(e)}")

        # Attendre un peu pour s'assurer que tout est envoyé
        await asyncio.sleep(1)
        logger.info("✅ Fonction notify_webhook terminée")

    except Exception as e:
        logger.error(f"❌ Erreur globale webhook: {str(e)}")
        logger.error("Traceback complet:", exc_info=True)

# Async wrapper for prompt processing
async def process_prompts_async(prompts, brand_name, clients):
    """Process prompts asynchronously"""
    logger.info(f"🔄 Process prompts async - {len(prompts)} prompts, {len(clients)} clients")
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
    logger.info(f"🔄 Process prompts sync démarré")

    results = {
        platform: {
            'responses': [],
            'total_mentions': 0,
            'unique_mentions': 0
        }
        for platform in clients.keys()
    }

    for idx, prompt in enumerate(prompts):
        logger.info(f"  Prompt {idx + 1}/{len(prompts)}: {prompt[:30]}...")

        for platform_key, client in clients.items():
            try:
                logger.info(f"    - Appel {platform_key}...")

                if platform_key == 'perplexity':
                    response, sources = client.call_api(prompt)
                    mentions = count_brand_mentions(response, brand_name)
                    logger.info(f"      ✓ {mentions} mentions, {len(sources)} sources")

                    results[platform_key]['responses'].append({
                        'prompt': prompt,
                        'response': response,
                        'mentions': mentions,
                        'sources': sources
                    })
                else:
                    response = client.call_api(prompt)
                    mentions = count_brand_mentions(response, brand_name)
                    logger.info(f"      ✓ {mentions} mentions")

                    results[platform_key]['responses'].append({
                        'prompt': prompt,
                        'response': response,
                        'mentions': mentions
                    })

                results[platform_key]['total_mentions'] += mentions
                if mentions > 0:
                    results[platform_key]['unique_mentions'] += 1

            except Exception as e:
                logger.error(f"      ✗ Erreur: {str(e)}")
                results[platform_key]['responses'].append({
                    'prompt': prompt,
                    'response': f"Error: {str(e)}",
                    'mentions': 0
                })

    logger.info("✅ Process prompts sync terminé")
    return results


def add_competitor_tracking(results, competitors):
    """Add competitor tracking to results"""
    logger.info(f"📊 Ajout tracking pour {len(competitors)} concurrents")

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
    logger.info("🧠 Démarrage analyse LLM async")
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
        logger.info("🧪 TEST ENDPOINT - Données reçues:")
        logger.info(json.dumps(data, indent=2)[:500])

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


# New debug endpoint
@app.post("/debug-api-keys")
async def debug_api_keys(request: Request):
    """Debug endpoint to check API keys reception"""
    try:
        data = await request.json()
        api_infos = data.get('data', {}).get('api_infos', {})

        result = {
            "timestamp": datetime.now().isoformat(),
            "api_keys_status": {
                "openai": {
                    "present": bool(api_infos.get('chatgpt_api_key')),
                    "length": len(api_infos.get('chatgpt_api_key', '')),
                    "starts_with": api_infos.get('chatgpt_api_key', '')[:10] + "..." if api_infos.get(
                        'chatgpt_api_key') else "N/A"
                },
                "perplexity": {
                    "present": bool(api_infos.get('perplexity_api_key')),
                    "length": len(api_infos.get('perplexity_api_key', '')),
                    "starts_with": api_infos.get('perplexity_api_key', '')[:10] + "..." if api_infos.get(
                        'perplexity_api_key') else "N/A"
                },
                "gemini": {
                    "present": bool(api_infos.get('gemini_api_key')),
                    "length": len(api_infos.get('gemini_api_key', '')),
                    "starts_with": api_infos.get('gemini_api_key', '')[:10] + "..." if api_infos.get(
                        'gemini_api_key') else "N/A"
                }
            },
            "form_data": {
                "brand": data.get('data', {}).get('form_infos', {}).get('brand_name'),
                "prompts_count": len(data.get('data', {}).get('form_infos', {}).get('prompts_list', []))
            }
        }

        logger.info("🔍 DEBUG API KEYS:")
        logger.info(json.dumps(result, indent=2))

        return result

    except Exception as e:
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


if __name__ == "__main__":
    import uvicorn

    # For local testing
    logger.info("🚀 Démarrage du serveur en mode local")
    uvicorn.run(app, host="0.0.0.0", port=8080)