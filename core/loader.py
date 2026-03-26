import os
from langchain_community.document_loaders import PyMuPDFLoader
from rapidocr_onnxruntime import RapidOCR

def smart_load_pdf(file_path):
    """
    智能加载：先尝试直接读取，如果没内容则启动 OCR
    """
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    
    # 检查是否每一页都几乎没文字 (判断是否为扫描件)
    total_text_len = sum(len(doc.page_content.strip()) for doc in docs)
    
    if total_text_len < 50:  # 如果整本书文字极少，判定为扫描件
        print(f"⚠️ 检测到扫描件: {os.path.basename(file_path)}，启动 OCR 识别...")
        engine = RapidOCR()
        ocr_docs = []
        
        # 使用 PyMuPDF 将每一页转为图片再识别
        import fitz # PyMuPDF
        doc = fitz.open(file_path)
        for i, page in enumerate(doc):
            pix = page.get_pixmap()
            img_bytes = pix.tobytes()
            result, _ = engine(img_bytes)
            
            if result:
                # 将 OCR 结果拼接成文本
                page_text = "\n".join([line[1] for line in result])
                # 保持元数据一致
                from langchain_core.documents import Document
                ocr_docs.append(Document(
                    page_content=page_text,
                    metadata={"source": file_path, "page": i + 1}
                ))
        return ocr_docs
    
    return docs