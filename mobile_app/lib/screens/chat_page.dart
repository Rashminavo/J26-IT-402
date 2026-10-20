import 'package:flutter/material.dart';

/// MindBot conversation screen shown from the dashboard's Chat destination.
class ChatPage extends StatefulWidget {
  const ChatPage({super.key});

  @override
  State<ChatPage> createState() => _ChatPageState();
}

class _ChatMessage {
  const _ChatMessage(this.text, {required this.fromBot, this.time = 'Just now'});

  final String text;
  final bool fromBot;
  final String time;
}

class _ChatPageState extends State<ChatPage> {
  static const _purple = Color(0xFF8275E7);
  static const _ink = Color(0xFF211C62);
  final _inputController = TextEditingController();
  final _scrollController = ScrollController();
  final List<_ChatMessage> _messages = [
    const _ChatMessage(
      "Hello Kamal! You look a little tired today. How are you feeling right now?",
      fromBot: true,
      time: '10:24 AM',
    ),
    const _ChatMessage(
      "Honestly, I've been focusing on work all morning.",
      fromBot: false,
      time: '10:24 AM',
    ),
    const _ChatMessage(
      "I'm sorry to hear that. Lack of focus can often be linked to the fatigue I'm noticing. Would you like to try a 2-minute breathing exercise or just talk it through?",
      fromBot: true,
      time: '10:25 AM',
    ),
    const _ChatMessage(
      "Let's talk. I think it's just stress about the upcoming deadline.",
      fromBot: false,
      time: '10:25 AM',
    ),
  ];

  @override
  void dispose() {
    _inputController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  void _sendMessage() {
    final text = _inputController.text.trim();
    if (text.isEmpty) return;
    setState(() {
      _messages.add(_ChatMessage(text, fromBot: false));
      _inputController.clear();
    });
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 250),
          curve: Curves.easeOut,
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        _header(context),
        Expanded(
          child: ListView.builder(
            controller: _scrollController,
            padding: const EdgeInsets.fromLTRB(18, 14, 18, 14),
            itemCount: _messages.length,
            itemBuilder: (context, index) => _messageBubble(_messages[index]),
          ),
        ),
        _sentimentCard(),
        _composer(),
      ],
    );
  }

  Widget _header(BuildContext context) => Container(
        padding: const EdgeInsets.fromLTRB(12, 6, 16, 9),
        decoration: const BoxDecoration(
          color: Color(0xFFF4F3FF),
          border: Border(bottom: BorderSide(color: Color(0xFFEAE8F7))),
        ),
        child: Row(
          children: [
            IconButton(
              tooltip: 'Back to home',
              onPressed: () => Navigator.of(context).maybePop(),
              icon: const Icon(Icons.arrow_back_ios_new_rounded, size: 17),
              color: _ink,
              visualDensity: VisualDensity.compact,
            ),
            const Spacer(),
            const Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text('MindBot', style: TextStyle(color: _ink, fontSize: 13, fontWeight: FontWeight.w800)),
                SizedBox(height: 2),
                Row(children: [
                  Icon(Icons.circle, size: 6, color: Color(0xFF5BCB91)),
                  SizedBox(width: 4),
                  Text('Online', style: TextStyle(color: Color(0xFF85849A), fontSize: 8)),
                ]),
              ],
            ),
            const Spacer(),
            IconButton(
              tooltip: 'Start video session',
              onPressed: () => ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text('Video sessions are coming soon.'), behavior: SnackBarBehavior.floating),
              ),
              icon: const Icon(Icons.videocam_rounded, size: 19),
              color: _ink,
              visualDensity: VisualDensity.compact,
            ),
          ],
        ),
      );

  Widget _messageBubble(_ChatMessage message) {
    final bot = message.fromBot;
    return Padding(
      padding: const EdgeInsets.only(bottom: 11),
      child: Row(
        mainAxisAlignment: bot ? MainAxisAlignment.start : MainAxisAlignment.end,
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          if (bot) ...[
            Container(
              width: 25,
              height: 25,
              decoration: const BoxDecoration(color: Colors.white, shape: BoxShape.circle),
              child: const Icon(Icons.smart_toy_rounded, size: 15, color: _purple),
            ),
            const SizedBox(width: 7),
          ],
          Flexible(
            child: Column(
              crossAxisAlignment: bot ? CrossAxisAlignment.start : CrossAxisAlignment.end,
              children: [
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                  decoration: BoxDecoration(
                    color: bot ? Colors.white : _purple,
                    borderRadius: BorderRadius.only(
                      topLeft: const Radius.circular(14),
                      topRight: const Radius.circular(14),
                      bottomLeft: Radius.circular(bot ? 4 : 14),
                      bottomRight: Radius.circular(bot ? 14 : 4),
                    ),
                    boxShadow: [BoxShadow(color: _ink.withValues(alpha: .035), blurRadius: 8, offset: const Offset(0, 3))],
                  ),
                  child: Text(
                    message.text,
                    style: TextStyle(color: bot ? const Color(0xFF454266) : Colors.white, fontSize: 10, height: 1.45),
                  ),
                ),
                const SizedBox(height: 3),
                Text(message.time, style: const TextStyle(color: Color(0xFFAAA8B8), fontSize: 7)),
              ],
            ),
          ),
          if (!bot) const SizedBox(width: 25),
        ],
      ),
    );
  }

  Widget _sentimentCard() => Container(
        margin: const EdgeInsets.fromLTRB(16, 0, 16, 9),
        padding: const EdgeInsets.fromLTRB(11, 9, 11, 8),
        decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(11)),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Row(children: [
              Text('LIVE SENTIMENT', style: TextStyle(color: _ink, fontSize: 8, fontWeight: FontWeight.w800, letterSpacing: .2)),
              Spacer(),
              Text('Slightly Anxious', style: TextStyle(color: Color(0xFF85849A), fontSize: 8)),
            ]),
            const SizedBox(height: 7),
            ClipRRect(
              borderRadius: BorderRadius.circular(5),
              child: const Row(children: [
                Expanded(flex: 35, child: SizedBox(height: 4, child: ColoredBox(color: Color(0xFF62C895)))),
                Expanded(flex: 45, child: SizedBox(height: 4, child: ColoredBox(color: Color(0xFFFFB454)))),
                Expanded(flex: 20, child: SizedBox(height: 4, child: ColoredBox(color: Color(0xFFE76577)))),
              ]),
            ),
            const SizedBox(height: 4),
            const Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
              Text('POS 35%', style: TextStyle(color: Color(0xFF9998A8), fontSize: 6)),
              Text('NEU 45%', style: TextStyle(color: Color(0xFF9998A8), fontSize: 6)),
              Text('NEG 20%', style: TextStyle(color: Color(0xFF9998A8), fontSize: 6)),
            ]),
          ],
        ),
      );

  Widget _composer() => Padding(
        padding: EdgeInsets.fromLTRB(16, 0, 16, MediaQuery.viewInsetsOf(context).bottom > 0 ? 8 : 10),
        child: Row(
          children: [
            Material(
              color: Colors.white,
              shape: const CircleBorder(),
              child: IconButton(
                tooltip: 'Voice input',
                onPressed: () => ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(content: Text('Voice input is coming soon.'), behavior: SnackBarBehavior.floating),
                ),
                icon: const Icon(Icons.mic_none_rounded, size: 18),
                color: const Color(0xFF85849A),
                constraints: const BoxConstraints.tightFor(width: 38, height: 38),
              ),
            ),
            const SizedBox(width: 7),
            Expanded(
              child: TextField(
                controller: _inputController,
                textInputAction: TextInputAction.send,
                onSubmitted: (_) => _sendMessage(),
                decoration: InputDecoration(
                  hintText: 'Type how you feel...',
                  hintStyle: const TextStyle(color: Color(0xFFA5A3B2), fontSize: 10),
                  filled: true,
                  fillColor: Colors.white,
                  contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 11),
                  border: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: BorderSide.none),
                ),
                style: const TextStyle(color: _ink, fontSize: 11),
              ),
            ),
            const SizedBox(width: 7),
            Material(
              color: _purple,
              borderRadius: BorderRadius.circular(11),
              child: IconButton(
                tooltip: 'Send message',
                onPressed: _sendMessage,
                icon: const Icon(Icons.send_rounded, size: 17),
                color: Colors.white,
                constraints: const BoxConstraints.tightFor(width: 38, height: 38),
              ),
            ),
          ],
        ),
      );
}
