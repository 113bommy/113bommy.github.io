/* /assets/js/_commons/pdf-renderer.js (CDN 버전으로 수정) */
document.addEventListener("DOMContentLoaded", function() {
  const pdfContainers = document.querySelectorAll('.pdf-container');
  if (pdfContainers.length === 0) {
    return;
  }

  const pdfjsVersion = '3.11.174'; // 안정적인 최신 버전 명시

  // PDF.js 라이브러리가 로드되었는지 확인
  if (typeof pdfjsLib === 'undefined') {
    let script = document.createElement('script');
    // 안정적인 CDN 주소로 변경
    script.src = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsVersion}/pdf.min.js`;
    document.head.appendChild(script);
    script.onload = () => {
      initializePdfRenderer();
    };
  } else {
    initializePdfRenderer();
  }

  function initializePdfRenderer() {
    // Worker 스크립트 경로 설정 (CDN 주소로 변경)
    pdfjsLib.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsVersion}/pdf.worker.min.js`;
    
    pdfContainers.forEach((container, index) => {
      const url = container.dataset.pdfUrl;
      renderAllPages(container, url);
    });
  }

  function renderAllPages(container, url) {
    pdfjsLib.getDocument(url).promise.then(function(pdfDoc) {
      // 렌더링 전에 기존 내용을 비워줌
      container.innerHTML = ''; 
      for(let pageNum = 1; pageNum <= pdfDoc.numPages; pageNum++) {
        renderPage(pdfDoc, pageNum, container);
      }
    });
  }

  function renderPage(pdfDoc, pageNum, container) {
    pdfDoc.getPage(pageNum).then(function(page) {
      const viewport = page.getViewport({scale: 1.5});
      const canvas = document.createElement('canvas');
      const context = canvas.getContext('2d');
      canvas.height = viewport.height;
      canvas.width = viewport.width;
      canvas.style.marginBottom = '20px'; // 페이지 간 여백 추가

      container.appendChild(canvas);

      const renderContext = {
        canvasContext: context,
        viewport: viewport
      };
      page.render(renderContext);
    });
  }
});