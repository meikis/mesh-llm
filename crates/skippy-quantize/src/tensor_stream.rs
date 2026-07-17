use std::io::Write;

use anyhow::{Context, Result, ensure};

use crate::float_convert::{FloatDType, convert_float_chunk};
use crate::hf_checkpoint::SafetensorFile;

#[derive(Debug, Clone, Copy)]
pub(crate) enum TensorTransform {
    Direct,
    AlternatingRows { parity: u64, row_elements: u64 },
}

pub(crate) struct TensorSegment {
    pub(crate) file_index: usize,
    pub(crate) source_name: String,
    pub(crate) source_dtype: FloatDType,
    pub(crate) target_dtype: FloatDType,
    pub(crate) element_count: u64,
    pub(crate) source_byte_len: u64,
    pub(crate) target_byte_len: u64,
    pub(crate) transform: TensorTransform,
}

pub(crate) fn stream_tensor_segment<W: Write>(
    writer: &mut W,
    file: &SafetensorFile,
    segment: &TensorSegment,
    buffer_size: usize,
) -> Result<u64> {
    match segment.transform {
        TensorTransform::Direct => stream_direct(writer, file, segment, buffer_size),
        TensorTransform::AlternatingRows {
            parity,
            row_elements,
        } => stream_alternating_rows(writer, file, segment, buffer_size, parity, row_elements),
    }
}

fn stream_direct<W: Write>(
    writer: &mut W,
    file: &SafetensorFile,
    segment: &TensorSegment,
    buffer_size: usize,
) -> Result<u64> {
    if segment.source_dtype == segment.target_dtype {
        let copied = file.stream_tensor(&segment.source_name, writer, buffer_size)?;
        ensure_source_bytes(segment, copied)?;
        return Ok(copied);
    }

    let element_size = usize::try_from(segment.source_dtype.byte_size())
        .context("source dtype byte size does not fit usize")?;
    let chunk_size = aligned_chunk_size(buffer_size, element_size);
    let mut output_bytes = 0_u64;
    let mut source_bytes = 0_u64;
    file.stream_tensor_chunks(&segment.source_name, chunk_size, |chunk| {
        ensure!(
            chunk.len() % element_size == 0,
            "chunk for {} split an element boundary",
            segment.source_name
        );
        source_bytes += chunk.len() as u64;
        output_bytes +=
            convert_float_chunk(chunk, segment.source_dtype, segment.target_dtype, writer)?;
        Ok(())
    })?;
    ensure_source_bytes(segment, source_bytes)?;
    ensure!(
        source_bytes / segment.source_dtype.byte_size() == segment.element_count,
        "read element count mismatch for {}",
        segment.source_name
    );
    Ok(output_bytes)
}

fn stream_alternating_rows<W: Write>(
    writer: &mut W,
    file: &SafetensorFile,
    segment: &TensorSegment,
    buffer_size: usize,
    parity: u64,
    row_elements: u64,
) -> Result<u64> {
    ensure!(parity < 2, "alternating-row parity must be zero or one");
    ensure!(row_elements > 0, "alternating-row width must be non-zero");
    let row_bytes = row_elements
        .checked_mul(segment.source_dtype.byte_size())
        .context("alternating-row byte length overflow")?;
    let row_bytes = usize::try_from(row_bytes).context("row byte length does not fit usize")?;
    let chunk_size = aligned_chunk_size(buffer_size, row_bytes);
    let mut source_bytes = 0_u64;
    let mut output_bytes = 0_u64;
    let mut row_index = 0_u64;
    file.stream_tensor_chunks(&segment.source_name, chunk_size, |chunk| {
        ensure!(
            chunk.len() % row_bytes == 0,
            "chunk for {} split a fused SwiGLU row",
            segment.source_name
        );
        source_bytes += chunk.len() as u64;
        for row in chunk.chunks_exact(row_bytes) {
            if row_index % 2 == parity {
                output_bytes +=
                    convert_float_chunk(row, segment.source_dtype, segment.target_dtype, writer)?;
            }
            row_index += 1;
        }
        Ok(())
    })?;
    ensure_source_bytes(segment, source_bytes)?;
    ensure!(
        output_bytes == segment.target_byte_len,
        "deinterleaved {} bytes for {}, expected {}",
        output_bytes,
        segment.source_name,
        segment.target_byte_len
    );
    Ok(output_bytes)
}

fn ensure_source_bytes(segment: &TensorSegment, copied: u64) -> Result<()> {
    ensure!(
        copied == segment.source_byte_len,
        "read {} bytes for {}, expected {}",
        copied,
        segment.source_name,
        segment.source_byte_len
    );
    Ok(())
}

fn aligned_chunk_size(buffer_size: usize, alignment: usize) -> usize {
    let aligned = buffer_size - (buffer_size % alignment);
    aligned.max(alignment)
}
