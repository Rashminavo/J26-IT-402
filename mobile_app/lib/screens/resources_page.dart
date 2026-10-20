import 'package:flutter/material.dart';
import '../theme/app_colors.dart';

class ResourcesPage extends StatefulWidget {
  const ResourcesPage({super.key});

  @override
  State<ResourcesPage> createState() => _ResourcesPageState();
}

class _ResourcesPageState extends State<ResourcesPage> {
  final _searchController = TextEditingController();
  int _category = 0;
  static const _categories = ['All', 'Articles', 'Exercises', 'Emergency'];
  static const _resources = [
    ('Box Breathing Technique', 'Learn how controlled breathing can help you feel calm in just a few minutes.', 'BREATHING', '5 min read', Icons.self_improvement_rounded, Color(0xFF4EBB9A)),
    ('Managing Dark Days', 'Small, supportive steps for when your mood feels low.', 'DEPRESSION', '7 min read', Icons.wb_twilight_rounded, Color(0xFF8A78DD)),
    ('Building Your Support Net', 'Ways to connect with people who care about you.', 'COMMUNITY', '6 min read', Icons.groups_rounded, Color(0xFFEF9D48)),
  ];

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  void _openResource(String title) {
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(SnackBar(content: Text('$title is coming soon.'), behavior: SnackBarBehavior.floating));
  }

  @override
  Widget build(BuildContext context) {
    final query = _searchController.text.trim().toLowerCase();
    final resources = _resources.where((item) => item.$1.toLowerCase().contains(query) || item.$2.toLowerCase().contains(query)).toList();
    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(24, 18, 24, 14),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        const Text('Resources', style: TextStyle(color: AppColors.ink, fontSize: 17, fontWeight: FontWeight.w800)),
        const SizedBox(height: 12),
        TextField(controller: _searchController, onChanged: (_) => setState(() {}), decoration: InputDecoration(hintText: 'Search for articles, exercises...', hintStyle: const TextStyle(color: Color(0xFFAAA9B5), fontSize: 10), prefixIcon: const Icon(Icons.search_rounded, color: Color(0xFF85849A), size: 17), filled: true, fillColor: Colors.white, contentPadding: const EdgeInsets.symmetric(vertical: 10), border: OutlineInputBorder(borderRadius: BorderRadius.circular(10), borderSide: BorderSide.none))),
        const SizedBox(height: 9),
        SizedBox(height: 30, child: ListView.separated(scrollDirection: Axis.horizontal, itemCount: _categories.length, separatorBuilder: (_, _) => const SizedBox(width: 7), itemBuilder: (context, index) {
          final selected = _category == index;
          return ChoiceChip(label: Text(_categories[index]), selected: selected, onSelected: (_) => setState(() => _category = index), labelStyle: TextStyle(fontSize: 9, color: selected ? Colors.white : const Color(0xFF77758C), fontWeight: FontWeight.w500), padding: const EdgeInsets.symmetric(horizontal: 5), visualDensity: VisualDensity.compact, showCheckmark: false, backgroundColor: Colors.white, selectedColor: AppColors.purple, side: BorderSide.none, shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)));
        })),
        if (_category == 0 || _category == 1) ...[const SizedBox(height: 9), _FeaturedResource(onTap: () => _openResource('Mastering Mindfulness in 7 Days'))],
        const SizedBox(height: 16),
        const Text('Recommended for you', style: TextStyle(color: AppColors.ink, fontSize: 12, fontWeight: FontWeight.w800)),
        const SizedBox(height: 8),
        if (resources.isEmpty) const Padding(padding: EdgeInsets.symmetric(vertical: 20), child: Center(child: Text('No resources found.', style: TextStyle(color: Color(0xFF85849A), fontSize: 11)))) else ...resources.map((item) => Padding(padding: const EdgeInsets.only(bottom: 8), child: _RecommendedResource(title: item.$1, description: item.$2, tag: item.$3, duration: item.$4, icon: item.$5, color: item.$6, onTap: () => _openResource(item.$1)))),
        const SizedBox(height: 3),
        InkWell(
          onTap: () => _openResource('Emergency support'),
          borderRadius: BorderRadius.circular(10),
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 9),
            decoration: BoxDecoration(color: const Color(0xFFFFF0F1), borderRadius: BorderRadius.circular(10)),
            child: const Row(children: [
              Icon(Icons.favorite_rounded, color: Color(0xFFFF5264), size: 14),
              SizedBox(width: 7),
              Expanded(child: Text('Need immediate help?', style: TextStyle(color: Color(0xFFE74456), fontSize: 9, fontWeight: FontWeight.w700))),
              Text('Call helpline', style: TextStyle(color: Colors.white, backgroundColor: Color(0xFFE74456), fontSize: 8, fontWeight: FontWeight.w700)),
            ]),
          ),
        ),
      ]),
    );
  }
}

class _FeaturedResource extends StatelessWidget {
  const _FeaturedResource({required this.onTap});
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.white,
      borderRadius: BorderRadius.circular(12),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              height: 104,
              decoration: const BoxDecoration(
                borderRadius: BorderRadius.vertical(top: Radius.circular(12)),
                gradient: LinearGradient(
                  colors: [Color(0xFF91BFEA), Color(0xFFDBA0BF), Color(0xFF243A65)],
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                ),
              ),
              child: Stack(
                children: [
                  const Positioned.fill(child: Icon(Icons.landscape_rounded, size: 70, color: Color(0x66454B78))),
                  Positioned(
                    left: 8,
                    top: 8,
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                      decoration: BoxDecoration(color: const Color(0xFFFFB53E), borderRadius: BorderRadius.circular(10)),
                      child: const Text('FEATURED', style: TextStyle(color: Colors.white, fontSize: 7, fontWeight: FontWeight.w800)),
                    ),
                  ),
                ],
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(10),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('Mastering Mindfulness in 7 Days', style: TextStyle(color: AppColors.ink, fontSize: 11, fontWeight: FontWeight.w800)),
                  const SizedBox(height: 4),
                  const Text('A comprehensive guide to starting your meditation journey and finding inner peace amidst daily chaos.', maxLines: 2, overflow: TextOverflow.ellipsis, style: TextStyle(color: Color(0xFF85849A), fontSize: 8, height: 1.4)),
                  const SizedBox(height: 8),
                  SizedBox(
                    width: double.infinity,
                    height: 31,
                    child: FilledButton(
                      onPressed: onTap,
                      style: FilledButton.styleFrom(backgroundColor: AppColors.purple, shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8))),
                      child: const Text('Read Now', style: TextStyle(fontSize: 9, fontWeight: FontWeight.w600)),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
class _RecommendedResource extends StatelessWidget {
  const _RecommendedResource({required this.title, required this.description, required this.tag, required this.duration, required this.icon, required this.color, required this.onTap});
  final String title, description, tag, duration;
  final IconData icon;
  final Color color;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) => Material(color: Colors.white, borderRadius: BorderRadius.circular(11), child: InkWell(onTap: onTap, borderRadius: BorderRadius.circular(11), child: SizedBox(height: 81, child: Row(children: [
        Container(width: 77, height: double.infinity, margin: const EdgeInsets.all(5), decoration: BoxDecoration(color: color.withValues(alpha: .15), borderRadius: BorderRadius.circular(8)), child: Icon(icon, color: color, size: 37)),
        Expanded(child: Padding(padding: const EdgeInsets.fromLTRB(4, 9, 9, 7), child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [Text(title, maxLines: 1, overflow: TextOverflow.ellipsis, style: const TextStyle(color: AppColors.ink, fontSize: 9, fontWeight: FontWeight.w800)), const SizedBox(height: 3), Expanded(child: Text(description, maxLines: 2, overflow: TextOverflow.ellipsis, style: const TextStyle(color: Color(0xFF85849A), fontSize: 7, height: 1.3))), Row(children: [Text(tag, style: TextStyle(color: color, fontSize: 6, fontWeight: FontWeight.w800)), const SizedBox(width: 8), const Icon(Icons.schedule_rounded, size: 8, color: Color(0xFF9998A8)), const SizedBox(width: 3), Text(duration, style: const TextStyle(color: Color(0xFF9998A8), fontSize: 7))])]))),
      ]))));
}
