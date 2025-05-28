use axum::{routing::get, Router};
use tower_http::cors::CorsLayer;

mod api;
mod core;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let app = Router::new()
        .route("/predict", axum::routing::post(api::handlers::predict))
        .route("/predict/batch", axum::routing::post(api::handlers::predict_batch))
        .route("/health", get(api::handlers::health))
        .route("/model/info", get(api::handlers::model_info))
        .layer(CorsLayer::permissive());

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8001").await?;
    println!("Rust API listening on http://0.0.0.0:8001");
    
    axum::serve(listener, app).await?;
    Ok(())
}